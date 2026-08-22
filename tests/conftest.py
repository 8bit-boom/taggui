import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QSettings

import utils.settings


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """
    Redirect `utils.settings.get_settings()` to a `QSettings` instance backed
    by a throwaway INI file, so tests never read or write the real user
    settings.
    """
    settings_path = str(tmp_path / 'settings.ini')

    def fake_qsettings(*_args, **_kwargs):
        return QSettings(settings_path, QSettings.Format.IniFormat)

    monkeypatch.setattr(utils.settings, 'QSettings', fake_qsettings)
    yield
