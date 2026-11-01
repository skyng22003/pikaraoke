import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from pikaraoke.routes.controller import controller_bp


@pytest.fixture
def app():
    """Create a Flask app for testing."""
    test_app = Flask(__name__)
    test_app.register_blueprint(controller_bp)
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


class TestControllerRoutes:
    """Tests for playback control routes."""

    @patch("pikaraoke.routes.controller.get_karaoke_instance")
    def test_play_arms_manual_start(self, mock_get_instance, client):
        """POST /api/play arms the waiting song without touching the player."""
        mock_karaoke = MagicMock()
        mock_get_instance.return_value = mock_karaoke

        response = client.post("/api/play")

        assert response.status_code == 200
        assert json.loads(response.data) == {"success": True}
        mock_karaoke.playback_controller.request_start.assert_called_once_with()

    @patch("pikaraoke.routes.controller.broadcast_event")
    @patch("pikaraoke.routes.controller.get_karaoke_instance")
    def test_audio_track_switches_and_stops_the_player(
        self, mock_get_instance, mock_broadcast, client
    ):
        mock_karaoke = MagicMock()
        mock_karaoke.playback_controller.now_playing_audio_tracks = ["Instrumental", "Vocals"]
        mock_karaoke.set_audio_track_current.return_value = True
        mock_get_instance.return_value = mock_karaoke

        response = client.post("/api/audio_track/1")

        assert json.loads(response.data) == {"success": True}
        mock_broadcast.assert_called_once_with("skip", "audio track change")
        mock_karaoke.set_audio_track_current.assert_called_once_with(1)

    @patch("pikaraoke.routes.controller.broadcast_event")
    @patch("pikaraoke.routes.controller.get_karaoke_instance")
    def test_audio_track_out_of_range_leaves_the_player_alone(
        self, mock_get_instance, mock_broadcast, client
    ):
        """Stopping the player for a track that does not exist would end the song."""
        mock_karaoke = MagicMock()
        mock_karaoke.playback_controller.now_playing_audio_tracks = ["Only track"]
        mock_get_instance.return_value = mock_karaoke

        response = client.post("/api/audio_track/1")

        assert response.status_code == 400
        mock_broadcast.assert_not_called()
        mock_karaoke.set_audio_track_current.assert_not_called()
