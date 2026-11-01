"""Unit tests for ffmpeg module."""

from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from pikaraoke.lib.ffmpeg import (
    AudioTrack,
    MediaInfo,
    build_ffmpeg_cmd,
    get_ffmpeg_version,
    is_ffmpeg_installed,
    is_transpose_enabled,
    probe_media,
    supports_hardware_h264_encoding,
)


class TestGetFfmpegVersion:
    """Tests for the get_ffmpeg_version function."""

    def test_version_parsed_correctly(self):
        """Test parsing FFmpeg version from output."""
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright (c) 2000-2022"

        with patch("subprocess.run", return_value=mock_result):
            result = get_ffmpeg_version()
            assert result == "5.1.2"

    def test_ffmpeg_not_installed(self):
        """Test handling when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_ffmpeg_version()
            assert result == "FFmpeg is not installed"

    def test_unable_to_parse_version(self):
        """Test handling when version can't be parsed."""
        mock_result = MagicMock()
        mock_result.stdout = "unexpected format"

        with patch("subprocess.run", return_value=mock_result):
            result = get_ffmpeg_version()
            assert result == "Unable to parse FFmpeg version"


class TestIsTransposeEnabled:
    """Tests for the is_transpose_enabled function."""

    def test_rubberband_available(self):
        """Test when rubberband filter is available."""
        mock_result = MagicMock()
        mock_result.stdout = b"... rubberband ... other filters"

        with patch("subprocess.run", return_value=mock_result):
            assert is_transpose_enabled() is True

    def test_rubberband_not_available(self):
        """Test when rubberband filter is not available."""
        mock_result = MagicMock()
        mock_result.stdout = b"aecho, aresample, volume"

        with patch("subprocess.run", return_value=mock_result):
            assert is_transpose_enabled() is False

    def test_ffmpeg_not_installed(self):
        """Test when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_transpose_enabled() is False


class TestBuildFfmpegCmdCdg:
    """Tests for where build_ffmpeg_cmd reads CDG graphics from."""

    def _fr(self, cdg_file_path=None, cdg_stream=None):
        fr = MagicMock()
        fr.file_path = "/songs/song.mp3"
        fr.file_extension = ".mp3"
        fr.cdg_file_path = cdg_file_path
        fr.cdg_stream = cdg_stream
        fr.output_file = "/tmp/1.m3u8"
        fr.init_filename = "1_init.mp4"
        fr.segment_pattern = "/tmp/1_segment_%03d.m4s"
        return fr

    @patch("pikaraoke.lib.ffmpeg.supports_hardware_h264_encoding", return_value=False)
    def test_reads_cdg_file(self, mock_hw):
        args = build_ffmpeg_cmd(self._fr(cdg_file_path="/songs/song.cdg")).get_args()
        assert "/songs/song.cdg" in args
        assert "pipe:" not in args

    @patch("pikaraoke.lib.ffmpeg.supports_hardware_h264_encoding", return_value=False)
    def test_reads_decoded_mcg_from_stdin(self, mock_hw):
        args = build_ffmpeg_cmd(self._fr(cdg_stream=b"cdg stream")).get_args()
        pipe = args.index("pipe:")
        assert args[pipe - 4 : pipe + 1] == ["-f", "cdg", "-copyts", "-i", "pipe:"]


class TestSupportsHardwareH264Encoding:
    """Tests for the supports_hardware_h264_encoding function."""

    def test_x86_returns_false(self):
        """Test that x86 architecture returns False."""
        with patch("platform.machine", return_value="x86_64"):
            assert supports_hardware_h264_encoding() is False

    def test_intel_returns_false(self):
        """Test that Intel architecture returns False."""
        with patch("platform.machine", return_value="i686"):
            assert supports_hardware_h264_encoding() is False

    def test_arm_with_encoder(self):
        """Test ARM with h264_v4l2m2m available."""
        mock_result = MagicMock()
        mock_result.stdout = b"h264_v4l2m2m encoder available"

        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", return_value=mock_result):
                assert supports_hardware_h264_encoding() is True

    def test_arm_without_encoder(self):
        """Test ARM without h264_v4l2m2m available."""
        mock_result = MagicMock()
        mock_result.stdout = b"libx264 encoder only"

        with patch("platform.machine", return_value="armv7l"):
            with patch("subprocess.run", return_value=mock_result):
                assert supports_hardware_h264_encoding() is False

    def test_arm_ffmpeg_not_found(self):
        """Test ARM when FFmpeg is not installed."""
        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert supports_hardware_h264_encoding() is False


class TestIsFfmpegInstalled:
    """Tests for the is_ffmpeg_installed function."""

    def test_ffmpeg_installed(self):
        """Test when FFmpeg is installed."""
        with patch("subprocess.run", return_value=MagicMock()):
            assert is_ffmpeg_installed() is True

    def test_ffmpeg_not_installed(self):
        """Test when FFmpeg is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert is_ffmpeg_installed() is False


class TestProbeMedia:
    """Tests for probe_media, which reads what playback needs before transcoding."""

    def _probe(self, result):
        with patch("pikaraoke.lib.ffmpeg.ffmpeg.probe", return_value=result):
            return probe_media("/songs/song.mkv")

    def test_returns_exact_duration(self):
        """Duration is returned unrounded: the CDG -t trim needs sub-second precision."""
        assert self._probe({"format": {"duration": "183.456"}}).duration == 183.456

    def test_missing_duration_is_none(self):
        assert self._probe({"format": {}}).duration is None

    def test_probe_error_returns_empty_info(self):
        with patch(
            "pikaraoke.lib.ffmpeg.ffmpeg.probe", side_effect=ffmpeg.Error("ffprobe", b"", b"")
        ):
            assert probe_media("/songs/broken.mkv") == MediaInfo()

    def test_lists_audio_tracks_named_by_title_then_language(self):
        info = self._probe(
            {
                "format": {"duration": "200"},
                "streams": [
                    {"codec_type": "audio", "codec_name": "ac3", "tags": {"title": "Instrumental"}},
                    {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "eng"}},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
            }
        )
        assert info.audio_tracks == [
            AudioTrack("Instrumental", "ac3"),
            AudioTrack("eng", "aac"),
            AudioTrack("", "aac"),
        ]

    def test_video_skips_attached_cover_art(self):
        info = self._probe(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "mjpeg",
                        "disposition": {"attached_pic": 1},
                    },
                    {"codec_type": "video", "codec_name": "hevc", "pix_fmt": "yuv420p10le"},
                ],
            }
        )
        assert (info.video_codec, info.video_pix_fmt) == ("hevc", "yuv420p10le")


class TestBuildFfmpegCmdVideoFile:
    """Tests for how build_ffmpeg_cmd maps and encodes a video file's streams."""

    def _fr(self, media, extension=".mkv"):
        fr = MagicMock()
        fr.file_path = f"/songs/song{extension}"
        fr.file_extension = extension
        fr.cdg_file_path = None
        fr.cdg_stream = None
        fr.media = media
        fr.output_file = "/tmp/1.m3u8"
        fr.init_filename = "1_init.mp4"
        fr.segment_pattern = "/tmp/1_segment_%03d.m4s"
        fr.variant_playlist = "/tmp/1_%v.m3u8"
        fr.variant_init_filename = "1_%v_init.mp4"
        fr.variant_segment_pattern = "/tmp/1_%v_segment_%03d.m4s"
        return fr

    def _args(self, media, **kwargs):
        with patch("pikaraoke.lib.ffmpeg.supports_hardware_h264_encoding", return_value=False):
            return build_ffmpeg_cmd(self._fr(media), normalize_audio=False, **kwargs).get_args()

    def _value(self, args, flag):
        return args[args.index(flag) + 1]

    def _two_tracks(self, video_codec="h264", pix_fmt="yuv420p", audio_codec="aac"):
        tracks = [AudioTrack("Instrumental", audio_codec), AudioTrack("Vocals", audio_codec)]
        return MediaInfo(video_codec=video_codec, video_pix_fmt=pix_fmt, audio_tracks=tracks)

    def test_maps_only_the_chosen_audio_track(self):
        """Mapping every track puts both into the stream, which the player cannot handle."""
        args = self._args(self._two_tracks(), audio_track=1)
        maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
        assert maps == ["0:a:1", "0:V:0"]

    def test_all_audio_tracks_become_switchable_hls_renditions(self):
        args = self._args(self._two_tracks(), audio_track=1, all_audio_tracks=True)
        maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
        assert maps == ["0:V:0", "0:a:0", "0:a:1"]
        assert self._value(args, "-var_stream_map") == (
            "v:0,agroup:aud,name:v a:0,agroup:aud,name:a0 a:1,agroup:aud,name:a1,default:yes"
        )
        assert self._value(args, "-master_pl_name") == "1.m3u8"
        assert args[-1] == "/tmp/1_%v.m3u8"

    def test_all_audio_tracks_filters_each_track(self):
        """Every rendition must play in the singer's key, whichever they switch to."""
        args = self._args(self._two_tracks(), semitones=2, all_audio_tracks=True)
        assert self._value(args, "-filter_complex").count("rubberband") == 2

    def test_copies_8bit_h264_from_any_container(self):
        args = self._args(self._two_tracks())
        assert self._value(args, "-vcodec") == "copy"
        assert "-pix_fmt" not in args

    def test_reencodes_10bit_video_to_8bit(self):
        """10-bit sources would otherwise become High 10 H.264, which browsers cannot play."""
        args = self._args(self._two_tracks(video_codec="hevc", pix_fmt="yuv420p10le"))
        assert self._value(args, "-vcodec") == "libx264"
        assert self._value(args, "-pix_fmt") == "yuv420p"

    def test_mp4_mode_copies_browser_audio(self):
        args = self._args(self._two_tracks(), force_mp4_encoding=True)
        assert self._value(args, "-acodec") == "copy"

    def test_mp4_mode_reencodes_ac3_audio(self):
        args = self._args(self._two_tracks(audio_codec="ac3"), force_mp4_encoding=True)
        assert self._value(args, "-acodec") == "aac"


class TestIsTransposeEnabledIndexError:
    """Additional tests for is_transpose_enabled IndexError handling."""

    def test_index_error_returns_false(self):
        """Test that IndexError returns False."""
        with patch("subprocess.run", side_effect=IndexError):
            assert is_transpose_enabled() is False


class TestSupportsHardwareH264EncodingIndexError:
    """Additional tests for supports_hardware_h264_encoding IndexError handling."""

    def test_index_error_returns_false(self):
        """Test that IndexError on ARM returns False."""
        with patch("platform.machine", return_value="aarch64"):
            with patch("subprocess.run", side_effect=IndexError):
                assert supports_hardware_h264_encoding() is False
