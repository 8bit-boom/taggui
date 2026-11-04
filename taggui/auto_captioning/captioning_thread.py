import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from PIL import UnidentifiedImageError
from PySide6.QtCore import QModelIndex, QThread, Qt, Signal

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.models_list import get_model_class
from models.image_list_model import ImageListModel
from utils.enums import CaptionDevice, CaptionPosition
from utils.image import Image
from utils.settings import get_tag_separator
from utils.utils import pluralize


def add_caption_to_tags(tags: list[str], caption: str,
                        caption_position: CaptionPosition) -> list[str]:
    if caption_position == CaptionPosition.DO_NOT_ADD or not caption:
        return tags
    tag_separator = get_tag_separator()
    new_tags = caption.split(tag_separator)
    # Make a copy of the tags so that the tags in the image list model are not
    # modified.
    tags = tags.copy()
    if caption_position == CaptionPosition.BEFORE_FIRST_TAG:
        tags[:0] = new_tags
    elif caption_position == CaptionPosition.AFTER_LAST_TAG:
        tags.extend(new_tags)
    elif caption_position == CaptionPosition.OVERWRITE_FIRST_TAG:
        if tags:
            tags[:1] = new_tags
        else:
            tags = new_tags
    elif caption_position == CaptionPosition.OVERWRITE_ALL_TAGS:
        tags = new_tags
    return tags


def format_duration(seconds: float) -> str:
    seconds_per_minute = 60
    seconds_per_hour = 60 * seconds_per_minute
    seconds_per_day = 24 * seconds_per_hour
    if seconds < seconds_per_minute:
        return f'{seconds:.1f} seconds'
    if seconds < seconds_per_hour:
        minutes = seconds / seconds_per_minute
        return f'{minutes:.1f} minutes'
    if seconds < seconds_per_day:
        hours = seconds / seconds_per_hour
        return f'{hours:.1f} hours'
    days = seconds / seconds_per_day
    return f'{days:.1f} days'


class ModelCacheHolder:
    """
    A per-device stand-in for the widget-level model cache, used by
    parallel multi-GPU captioning: each GPU gets its own loaded model
    cached here instead of all of them fighting over the widget's single
    `model`/`processor` slots (which are what `unload` and the single-GPU
    reload logic rely on).
    """

    def __init__(self):
        self.processor = None
        self.model = None
        self.model_id: str | None = None
        self.model_device_type: str | None = None
        self.is_model_loaded_in_4_bit = None
        self.is_model_loaded_in_8_bit = None
        self.is_model_cpu_offloaded = None
        self.is_model_using_sage_attention = None
        self.is_model_torch_compiled = None
        self.is_model_attention_implementation = None
        self.is_wd_tagger_loaded_in_8_bit = None


class CaptioningThread(QThread):
    text_outputted = Signal(str)
    clear_console_text_edit_requested = Signal()
    # The image index, the caption, and the tags with the caption added. The
    # third parameter must be declared as `list` instead of `list[str]` for it
    # to work.
    caption_generated = Signal(QModelIndex, str, list)
    progress_bar_update_requested = Signal(int)
    progress_bar_range_update_requested = Signal(int)
    progress_bar_format_update_requested = Signal(str)
    # Emitted when a test run ("caption the first N images, then pause")
    # reaches its count while more images remain. The widget asks the user
    # whether to continue and sets `continue_after_pilot`.
    pilot_run_finished_requested = Signal(int)

    def __init__(self, parent, image_list_model: ImageListModel,
                 selected_image_indices: list[QModelIndex],
                 caption_settings: dict, tag_separator: str,
                 models_directory_path: Path | None):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.selected_image_indices = selected_image_indices
        self.caption_settings = caption_settings
        self.tag_separator = tag_separator
        self.models_directory_path = models_directory_path
        self.is_error = False
        self.is_canceled = False
        # Reliability/progress state, shared between the main loop and any
        # parallel per-GPU workers; guarded by `progress_lock` where noted.
        self.progress_lock = threading.Lock()
        self.captioned_total = 0
        self.failed_images: list[tuple[str, str]] = []
        self.failed_image_indices: list[QModelIndex] = []
        self.skipped_tagged_count = 0
        self.skipped_unreadable_count = 0
        self.total_target_count = 0
        self.is_parallel = False
        self.pilot_run_count = 0
        self.pilot_run_prompted = False
        self.skip_already_captioned = False
        # Set by the widget after the test-run prompt; `None` = not asked.
        self.continue_after_pilot: bool | None = None

    def record_failure(self, image_index: QModelIndex, image: Image | None,
                       error_text: str):
        name = (image.path.name if image is not None
                else f'row {image_index.row()}')
        with self.progress_lock:
            self.failed_images.append((name, error_text))
            self.failed_image_indices.append(image_index)
        print(f'ERROR captioning {name}: {error_text}')

    def prepare_image(self, model: AutoCaptioningModel,
                      image_index: QModelIndex) -> tuple:
        """
        The CPU-side work for one image, run in a background thread ahead of
        time so that it overlaps with the GPU generation (or the in-flight
        HTTP request for remote endpoints) of the previous images. Returns
        `(status, image_index, image, image_prompt, payload)`, where
        `payload` is the prepared model inputs for per-image models, or the
        decoded PIL image for batched models (whose processor call must
        happen once for the whole batch, in the captioning thread). Failures
        are returned as a status instead of raised, so that one bad image
        doesn't end the run.
        """
        try:
            image: Image = self.image_list_model.data(
                image_index, Qt.ItemDataRole.UserRole)
            image_prompt = model.get_image_prompt(image)
            if model.supports_batching:
                payload = model.load_image(image)
            else:
                payload = model.get_model_inputs(image_prompt, image)
            return 'ok', image_index, image, image_prompt, payload
        except UnidentifiedImageError:
            return 'unreadable', image_index, None, None, None
        except Exception as exception:
            return ('failed', image_index, None, None,
                    f'{type(exception).__name__}: {exception}')

    def generate_single_result(
            self, model: AutoCaptioningModel,
            item: tuple) -> tuple[str | None, str, float]:
        """
        Caption one prepared image, isolating any error so that a single
        bad image (or a transient CUDA out-of-memory) doesn't end the run.
        Returns `(caption, console_output_caption, elapsed_seconds)`, with
        `caption = None` on failure.
        """
        _, image, image_prompt, payload = item
        start_time = perf_counter()
        try:
            if model.supports_batching:
                # The prepared payload is a PIL image; the per-image
                # processor call happens here.
                model_inputs = model.get_model_inputs(image_prompt, image)
            else:
                model_inputs = payload
            caption, console_caption = model.generate_caption(model_inputs,
                                                              image_prompt)
            return caption, console_caption, perf_counter() - start_time
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # One retry at half the pixel budget: an image that doesn't
            # fit alongside its KV cache often fits at a lower resolution.
            reduced_inputs = model.prepare_reduced_inputs(image_prompt,
                                                          image, 0.5)
            if reduced_inputs is None:
                return (None, 'torch.cuda.OutOfMemoryError (this model '
                              'does not support the reduced-resolution '
                              'retry)', perf_counter() - start_time)
            print(f'Ran out of GPU memory on {image.path.name}; retrying '
                  f'once at half the image size...')
            try:
                caption, console_caption = model.generate_caption(
                    reduced_inputs, image_prompt)
                return caption, console_caption, perf_counter() - start_time
            except Exception as exception:
                return (None, f'{type(exception).__name__}: {exception}',
                        perf_counter() - start_time)
        except Exception as exception:
            return (None, f'{type(exception).__name__}: {exception}',
                    perf_counter() - start_time)

    def generate_batch_results(
            self, model: AutoCaptioningModel,
            batch: list[tuple]) -> list[tuple[str | None, str, float]]:
        """
        Generate captions for one batch of prepared images. Returns
        `(caption, console_output_caption, elapsed_seconds)` per image, in
        the same order as `batch`; a `None` caption marks that image as
        failed.
        """
        batch_start_time = perf_counter()
        if model.is_remote_endpoint:
            # The server does its own batching, so "batching" here means
            # having several requests in flight at once. Each request is
            # timed individually and isolated: one failed request doesn't
            # cancel the others in flight.
            def request_one(item: tuple) -> tuple:
                request_start_time = perf_counter()
                try:
                    caption, console_caption = model.generate_caption(
                        item[3], item[2])
                    return (caption, console_caption,
                            perf_counter() - request_start_time)
                except Exception as exception:
                    return (None,
                            f'{type(exception).__name__}: {exception}',
                            perf_counter() - request_start_time)

            with ThreadPoolExecutor(max_workers=model.batch_size) as executor:
                return list(executor.map(request_one, batch))
        if model.supports_batching:
            batch_items = [{'image_prompt': item[2], 'pil_image': item[3],
                            'image': item[1]} for item in batch]
            try:
                model_inputs = model.get_batched_model_inputs(batch_items)
                captions, console_captions = model.generate_captions_batch(
                    model_inputs, batch_items)
                batch_elapsed = perf_counter() - batch_start_time
                batch_elapsed_per_image = batch_elapsed / len(batch)
                return [(caption, console_caption, batch_elapsed_per_image)
                        for caption, console_caption
                        in zip(captions, console_captions)]
            except torch.cuda.OutOfMemoryError:
                # The inputs of a batched call cannot be unpadded into
                # per-image inputs, so each image is re-prepared and
                # captioned on its own (with the OOM retry and error
                # isolation of the single-image path).
                torch.cuda.empty_cache()
                print(f'Ran out of GPU memory while captioning '
                      f'{len(batch)} images at once. Retrying them one at '
                      f'a time...')
                return [self.generate_single_result(model, item)
                        for item in batch]
            except Exception as exception:
                print(f'Batched captioning failed '
                      f'({type(exception).__name__}: {exception}). '
                      f'Retrying the images one at a time...')
                return [self.generate_single_result(model, item)
                        for item in batch]
        # A model without batching support: `prepare_image` already built
        # the per-image model inputs.
        return [self.generate_single_result(model, item) for item in batch]

    def process_batch(self, model: AutoCaptioningModel, batch: list[tuple],
                      caption_position: CaptionPosition,
                      are_multiple_images_selected: bool,
                      batch_start_time: float) -> int:
        """
        Generate captions for one batch, emit them into the image list, and
        print them to the console. Failed images are recorded and skipped.
        Returns the number of images captioned in this batch.
        """
        results = self.generate_batch_results(model, batch)
        successes = 0
        failures = 0
        for item, (caption, console_caption, elapsed) in zip(batch, results):
            image_index, image = item[0], item[1]
            if caption is None:
                failures += 1
                self.record_failure(image_index, image,
                                    console_caption or 'Unknown error')
                continue
            successes += 1
            tags = add_caption_to_tags(image.tags, caption, caption_position)
            self.caption_generated.emit(image_index, caption, tags)
            if console_caption is None:
                console_caption = caption
            if len(batch) == 1 and not model.is_remote_endpoint:
                print(f'{image.path.name} ({elapsed:.1f} s):\n'
                      f'{console_caption}')
            else:
                print(f'{image.path.name}:\n{console_caption}')
        batch_elapsed = perf_counter() - batch_start_time
        if len(batch) > 1 or model.is_remote_endpoint:
            failure_text = (f' {failures} '
                            f'{pluralize("image", failures)} failed.'
                            if failures else '')
            print(f'Captioned {successes} of {len(batch)} '
                  f'{pluralize("image", len(batch))} in {batch_elapsed:.1f} s'
                  f' ({batch_elapsed / len(batch):.1f} s/image).'
                  f'{failure_text}')
        with self.progress_lock:
            self.captioned_total += successes
            progress_value = self.captioned_total
        if are_multiple_images_selected:
            self.progress_bar_update_requested.emit(progress_value)
            self.update_progress_bar_format(model)
        return successes

    def update_progress_bar_format(self, model: AutoCaptioningModel):
        """
        Update the progress bar text with the ETA, throughput, and free
        VRAM, so the user can see what the speed knobs (batch size, pixel
        budget, quantization) are actually doing while a long run is going.
        """
        with self.progress_lock:
            captioned_total = self.captioned_total
            failed_count = len(self.failed_images)
            skipped_count = (self.skipped_tagged_count
                             + self.skipped_unreadable_count)
        processed_count = captioned_total + failed_count + skipped_count
        remaining_count = max(0, self.total_target_count - processed_count)
        elapsed = perf_counter() - self.captioning_start_time
        stats_text = ''
        if captioned_total > 0 and remaining_count > 0:
            seconds_per_image = elapsed / captioned_total
            eta_seconds = seconds_per_image * remaining_count
            images_per_hour = 3600 / seconds_per_image
            stats_text = (f' - ETA {format_duration(eta_seconds)}, '
                          f'{images_per_hour:.0f} images/hour')
        vram_text = ''
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                vram_text = (f' - {free_bytes / 1e9:.0f}/'
                             f'{total_bytes / 1e9:.0f} GB VRAM free')
            except Exception:
                pass
        self.progress_bar_format_update_requested.emit(
            f'{captioned_total} / {self.total_target_count} images captioned'
            f'{stats_text}{vram_text}')

    def _run_captioning_loop(self, model: AutoCaptioningModel,
                             image_indices: list[QModelIndex],
                             caption_position: CaptionPosition,
                             are_multiple_images_selected: bool):
        """
        The core captioning loop for one model (the single-device case, or
        one worker of a parallel multi-GPU run): overlaps CPU-side image
        preparation with generation and captions `batch_size` images per
        `generate()` call for models that support it.
        """
        batch_size = model.batch_size if (model.supports_batching
                                          or model.is_remote_endpoint) else 1
        # Keep a small pipeline of already-prepared images ahead of the
        # generation; a window much larger than this would start decoding
        # images that will be canceled anyway.
        max_pending_preps = max(4, batch_size * 2)
        remaining_indices = deque(image_indices)
        pending_preps: deque = deque()
        prep_executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix='caption-prep')

        def submit_preps():
            while (remaining_indices and not self.is_canceled
                   and len(pending_preps) < max_pending_preps):
                image_index = remaining_indices.popleft()
                pending_preps.append(prep_executor.submit(
                    self.prepare_image, model, image_index))

        try:
            submit_preps()
            local_captioned = 0
            while pending_preps and not self.is_canceled:
                # Collect (up to) one batch of prepared images, in order.
                batch = []
                while pending_preps and len(batch) < batch_size:
                    submit_preps()
                    future = pending_preps.popleft()
                    status, image_index, image, image_prompt, payload = (
                        future.result())
                    if status == 'unreadable':
                        with self.progress_lock:
                            self.skipped_unreadable_count += 1
                        print('Skipped an image because its file format is '
                              'not supported or it is a corrupted image.')
                        continue
                    if status == 'failed':
                        self.record_failure(image_index, None, payload)
                        continue
                    batch.append((image_index, image, image_prompt, payload))
                if not batch:
                    break
                if self.is_canceled:
                    break
                batch_start_time = perf_counter()
                local_captioned += self.process_batch(
                    model, batch, caption_position,
                    are_multiple_images_selected, batch_start_time)
                # Test run: pause after the configured number of images and
                # ask the user (via the widget) whether to continue, once
                # per run. Only the single-device run supports this (a
                # parallel run has no single point where all workers can
                # pause cleanly).
                if (self.pilot_run_count and not self.is_parallel
                        and not self.pilot_run_prompted
                        and local_captioned >= self.pilot_run_count
                        and (pending_preps or remaining_indices)):
                    self.pilot_run_prompted = True
                    print(f'Test run complete ({local_captioned} '
                          f'{pluralize("image", local_captioned)}). '
                          f'Waiting for your decision...')
                    self.pilot_run_finished_requested.emit(local_captioned)
                    while (self.continue_after_pilot is None
                           and not self.is_canceled):
                        self.msleep(100)
                    if self.is_canceled or not self.continue_after_pilot:
                        print('Stopped after the test run.')
                        return
        finally:
            # Pending (not-yet-consumed) preps are canceled; a prep already
            # running is awaited briefly by `wait=False` + process exit.
            prep_executor.shutdown(wait=False, cancel_futures=True)

    def caption_images(self, models: list[AutoCaptioningModel],
                       selected_image_indices: list[QModelIndex],
                       caption_position: CaptionPosition,
                       are_multiple_images_selected: bool):
        """
        Caption the selected images with the given model(s): one model for
        the single-device run, or one per GPU for a parallel run (the
        indices are then split round-robin across the GPUs, each with its
        own independent captioning loop).
        """
        indices = list(selected_image_indices)
        self.total_target_count = len(indices)
        if self.skip_already_captioned:
            kept_indices = []
            for index in indices:
                try:
                    image = self.image_list_model.data(
                        index, Qt.ItemDataRole.UserRole)
                except Exception:
                    image = None
                if image is not None and image.tags:
                    with self.progress_lock:
                        self.skipped_tagged_count += 1
                else:
                    kept_indices.append(index)
            skipped_count = self.skipped_tagged_count
            if skipped_count:
                print(f'Skipping {skipped_count} '
                      f'{pluralize("image", skipped_count)} that already '
                      f'have tags. (Untag them first if you want them '
                      f're-captioned.)')
                indices = kept_indices
                self.total_target_count = len(indices)
                if are_multiple_images_selected and indices:
                    self.progress_bar_range_update_requested.emit(
                        len(indices))
                if not indices:
                    print('Nothing left to caption.')
                    return
        if self.is_parallel:
            parallel_gpu_count = len(models)
            gpu_index_lists = [indices[worker_index::parallel_gpu_count]
                               for worker_index in range(parallel_gpu_count)]

            def run_worker(model: AutoCaptioningModel,
                           worker_indices: list[QModelIndex]):
                try:
                    self._run_captioning_loop(model, worker_indices,
                                              caption_position,
                                              are_multiple_images_selected)
                except Exception as exception:
                    print(f'Captioning on {model.device} failed: '
                          f'{type(exception).__name__}: {exception}')

            with ThreadPoolExecutor(
                    max_workers=parallel_gpu_count) as executor:
                list(executor.map(run_worker, models, gpu_index_lists))
        else:
            self._run_captioning_loop(models[0], indices, caption_position,
                                      are_multiple_images_selected)

    def run_captioning(self):
        model_id = self.caption_settings['model_id']
        model_class = get_model_class(model_id)
        device_setting = self.caption_settings['device']
        self.is_parallel = (device_setting == CaptionDevice.GPUS_PARALLEL
                            and torch.cuda.is_available()
                            and torch.cuda.device_count() >= 2)
        if (device_setting == CaptionDevice.GPUS_PARALLEL
                and not self.is_parallel):
            print('`Multiple GPUs (parallel)` needs at least two CUDA '
                  'GPUs; captioning on one device.')
        self.pilot_run_count = max(
            0, int(self.caption_settings.get('test_run_image_count', 0) or 0))
        self.skip_already_captioned = bool(
            self.caption_settings.get('skip_already_captioned', False))
        self.captioning_start_time = perf_counter()
        selected_image_indices = list(self.selected_image_indices)
        selected_image_count = len(selected_image_indices)
        are_multiple_images_selected = selected_image_count > 1
        if self.is_parallel:
            # One independent model instance per GPU (quantization, Sage,
            # etc. apply per instance); each is cached in its own holder so
            # the widget's single-model cache isn't involved.
            gpu_offset = self.caption_settings.get('gpu_index', 0)
            models = []
            for gpu_index in range(gpu_offset, torch.cuda.device_count()):
                device_settings = dict(self.caption_settings)
                device_settings['gpu_index'] = gpu_index
                model = model_class(captioning_thread_=self,
                                    caption_settings=device_settings)
                error_message = model.get_error_message()
                if error_message:
                    self.is_error = True
                    self.clear_console_text_edit_requested.emit()
                    print(error_message)
                    return
                models.append(model)
            self.clear_console_text_edit_requested.emit()
            captioning_start_datetime = datetime.now()
            print(f'Captioning on {len(models)} '
                  f'{pluralize("GPU", len(models))} in parallel (start '
                  f'time: {model.get_captioning_start_datetime_string(captioning_start_datetime)})')
            for model in models:
                print(f'Loading the model for {model.device}...')
                try:
                    model.load_processor_and_model()
                    model.monkey_patch_after_loading()
                except Exception as exception:
                    self.is_error = True
                    print(f'Failed to load the model for {model.device}: '
                          f'{type(exception).__name__}: {exception}')
                    raise
            if self.is_canceled:
                print('Canceled captioning.')
                return
            try:
                self.caption_images(models, selected_image_indices,
                                    self.caption_settings['caption_position'],
                                    are_multiple_images_selected)
            finally:
                # Free the per-GPU models; the parallel run doesn't use the
                # widget's model cache, so nothing here is kept for reuse.
                for model in models:
                    model.processor = None
                    model.model = None
                models.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            model: AutoCaptioningModel = model_class(
                captioning_thread_=self, caption_settings=self.caption_settings)
            error_message = model.get_error_message()
            if error_message:
                self.is_error = True
                self.clear_console_text_edit_requested.emit()
                print(error_message)
                return
            model.load_processor_and_model()
            model.monkey_patch_after_loading()
            if self.is_canceled:
                print('Canceled captioning.')
                return
            self.clear_console_text_edit_requested.emit()
            captioning_start_datetime = datetime.now()
            captioning_message = model.get_captioning_message(
                are_multiple_images_selected, captioning_start_datetime)
            print(captioning_message)
            if model.batch_size > 1 and not (model.supports_batching
                                             or model.is_remote_endpoint):
                print(f'{type(model).__name__} does not support captioning '
                      f'multiple images at once, so `Batch size` is ignored '
                      f'and images are captioned one at a time.')
            self.caption_images([model], selected_image_indices,
                                self.caption_settings['caption_position'],
                                are_multiple_images_selected)
        if self.is_canceled:
            print('Canceled captioning.')
            return
        if are_multiple_images_selected and self.captioned_total > 0:
            total_elapsed = perf_counter() - self.captioning_start_time
            print(f'Finished captioning {self.captioned_total} '
                  f'{pluralize("image", self.captioned_total)} in '
                  f'{format_duration(total_elapsed)} '
                  f'({total_elapsed / self.captioned_total:.1f} s/image) at '
                  f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
        if self.failed_images:
            failed_count = len(self.failed_images)
            print(f'{failed_count} of {self.total_target_count} '
                  f'{pluralize("image", self.total_target_count)} failed:')
            for name, error_text in self.failed_images[:20]:
                print(f'  {name}: {error_text}')
            if failed_count > 20:
                print(f'  ... and {failed_count - 20} more')
            print('The failed images are selected in the image list; '
                  'press `Start Auto-Captioning` again to retry just them.')
        if self.captioned_total == 0 and self.failed_images:
            self.is_error = True

    def run(self):
        try:
            self.run_captioning()
        except Exception as exception:
            self.is_error = True
            # Show the error message in the console text edit.
            raise exception

    def write(self, text: str):
        self.text_outputted.emit(text)

    def isatty(self) -> bool:
        return False
