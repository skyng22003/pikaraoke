"""FFmpeg utilities for media processing and transcoding."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ffmpeg

from pikaraoke.lib.get_platform import is_running_in_docker

if TYPE_CHECKING:
    from pikaraoke.lib.file_resolver import FileResolver


# Codecs every supported browser decodes, so they can be copied without re-encoding.
# 10-bit H.264 (High 10) is the common miss in MKV releases: browsers cannot play it.
_BROWSER_VIDEO = {("h264", "yuv420p"), ("h264", "yuvj420p")}
_BROWSER_AUDIO = {"aac", "mp3"}


@dataclass
class AudioTrack:
    """One audio stream of a media file, as offered to the singer.

    Attributes:
        name: The stream's title or language tag, empty when it has neither.
        codec: Codec name, used to decide whether the stream can be copied.
    """

    name: str
    codec: str | None


@dataclass
class MediaInfo:
    """What playback needs to know about a media file before transcoding it."""

    duration: float | None = None
    video_codec: str | None = None
    video_pix_fmt: str | None = None
    audio_tracks: list[AudioTrack] = field(default_factory=list)

    @property
    def is_browser_video(self) -> bool:
        return (self.video_codec, self.video_pix_fmt) in _BROWSER_VIDEO


def probe_media(file_path: str) -> MediaInfo:
    """Probe a media file's duration, video codec and audio tracks.

    Karaoke MKVs often carry several audio tracks (instrumental, with vocals),
    and cover art as an extra video stream, so streams are read individually.
    Returns an empty MediaInfo if the file cannot be probed.
    """
    try:
        probe = ffmpeg.probe(file_path)
    except (ffmpeg.Error, OSError) as e:
        logging.warning(f"Could not probe {file_path}: {e}")
        return MediaInfo()

    info = MediaInfo()
    duration = probe.get("format", {}).get("duration")
    info.duration = float(duration) if duration is not None else None
    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            tags = stream.get("tags", {})
            name = tags.get("title") or tags.get("language") or ""
            info.audio_tracks.append(AudioTrack(name=name, codec=stream.get("codec_name")))
        elif (
            codec_type == "video"
            and info.video_codec is None
            and not stream.get("disposition", {}).get("attached_pic")
        ):
            info.video_codec = stream.get("codec_name")
            info.video_pix_fmt = stream.get("pix_fmt")
    return info


def _process_audio(audio: Any, semitones: int, normalize_audio: bool, avsync: float) -> Any:
    """Apply sync adjustment, pitch shift and loudness normalization to an audio stream."""
    # Audio sync adjustment: delay or trim
    if avsync > 0:
        audio = audio.filter("adelay", f"{avsync * 1000}|{avsync * 1000}")
    elif avsync < 0:
        audio = audio.filter("atrim", start=-avsync)

    # Pitch shifting: 2^(semitones/12)
    if semitones != 0:
        audio = audio.filter("rubberband", pitch=2 ** (semitones / 12))

    # Loudness normalization
    if normalize_audio:
        audio = audio.filter("loudnorm", i=-16, tp=-1.5, lra=11)
    return audio


def build_ffmpeg_cmd(
    fr: FileResolver,
    semitones: int = 0,
    normalize_audio: bool = True,
    force_mp4_encoding: bool = False,
    buffer_fully_before_playback: bool = False,
    avsync: float = 0,
    cdg_pixel_scaling: bool = False,
    audio_track: int = 0,
    all_audio_tracks: bool = False,
) -> Any:
    """Build an ffmpeg command for transcoding media.

    Handles video/audio codec selection, pitch shifting, audio normalization,
    and CDG file rendering.

    Args:
        fr: FileResolver instance with source file information.
        semitones: Number of semitones to shift pitch (0 = no shift).
        normalize_audio: Whether to apply loudness normalization.
        force_mp4_encoding: If True, force mp4 encoding.
        avsync: Audio/video sync adjustment in seconds.
        cdg_pixel_scaling: Enable pixel scaling for CDG rendering.
        audio_track: Index of the audio stream to play, among the file's audio streams.
        all_audio_tracks: HLS only. Write every audio stream as its own rendition, so
            the player can switch between them without restarting the video.

    Returns:
        ffmpeg stream object ready to execute with run_async().
    """
    avsync = float(avsync)
    is_cdg = fr.cdg_file_path is not None or fr.cdg_stream is not None
    is_transposed = semitones != 0

    if fr.file_path is None:
        raise ValueError("File path is required to build ffmpeg command")

    # Use h/w acceleration on Pi
    using_hardware_encoder = supports_hardware_h264_encoding()
    default_vcodec = "h264_v4l2m2m" if using_hardware_encoder else "libx264"

    # CDG always needs encoding. Other video is copied only when it is already
    # browser-playable 8-bit H.264, whatever its container; VP8/VP9, HEVC and
    # 10-bit H.264 are re-encoded.
    if is_cdg:
        vcodec = "libx264"
    else:
        vcodec = "copy" if fr.media.is_browser_video else default_vcodec

    # Optimize bitrate: CDG is simple graphics (500k), video files need more
    # Pi 3B+ struggles with 5M in real-time, 2M provides better stability
    if is_cdg:
        vbitrate = "500k"
    elif using_hardware_encoder:
        vbitrate = "2M"
    else:
        vbitrate = "5M"

    # Copy audio if no processing is needed and browsers can decode it (MKVs often
    # carry AC3/DTS), otherwise re-encode with AAC. CDG always re-encodes audio.
    tracks = fr.media.audio_tracks
    track_codec = tracks[audio_track].codec if audio_track < len(tracks) else None
    needs_audio_processing = is_cdg or is_transposed or normalize_audio or avsync != 0
    acodec = "copy" if not needs_audio_processing and track_codec in _BROWSER_AUDIO else "aac"

    # For container formats with VFR or timestamp issues, use genpts
    if fr.file_extension in [".webm", ".avi", ".mov", ".mkv"]:
        input = ffmpeg.input(fr.file_path, **{"fflags": "+genpts"})
    else:
        input = ffmpeg.input(fr.file_path)
    # Map audio streams individually: muxing several into one rendition puts
    # every track of a multi-track MKV together, which the player cannot handle.
    track_indexes = range(len(tracks)) if all_audio_tracks else [audio_track]
    audios = [
        _process_audio(input[f"a:{i}"], semitones, normalize_audio, avsync) for i in track_indexes
    ]

    # Video source: CDG input or original video stream
    if is_cdg:
        logging.info("Playing CDG/MP3 file: " + fr.file_path)
        if fr.cdg_stream is not None:
            # Decoded from an .mcg; the caller writes it to ffmpeg's stdin
            cdg_input = ffmpeg.input("pipe:", f="cdg", copyts=None)
        else:
            cdg_input = ffmpeg.input(fr.cdg_file_path, copyts=None)
        video = cdg_input.video.filter("fps", fps=25)
        # CDG graphics end 7-17s before the music, and Chromium 150+ stops at the
        # shorter track rather than playing through, so "ended" never fires (#942).
        # 600s is a ceiling on any outro; -t trims back to the audio length.
        video = video.filter("tpad", stop_mode="clone", stop_duration=600)
        if cdg_pixel_scaling:
            video = video.filter("scale", -1, 720, flags="neighbor")
    else:
        # Capital V skips attached pictures (cover art), which MKVs often carry.
        video = input["V:0"]

    # Encoded video is forced to 8-bit 4:2:0, as 10-bit sources otherwise
    # produce High 10 H.264, which browsers cannot decode.
    video_opts = {} if vcodec == "copy" else {"pix_fmt": "yuv420p"}
    if is_cdg and fr.duration_exact:
        video_opts["t"] = fr.duration_exact + avsync

    # Build output based on format
    if force_mp4_encoding:
        movflags = (
            "+faststart" if buffer_fully_before_playback else "frag_keyframe+default_base_moof"
        )
        output = ffmpeg.output(
            audios[0],
            video,
            fr.output_file,
            vcodec=vcodec,
            acodec=acodec,
            preset="ultrafast",
            listen=1,
            f="mp4",
            video_bitrate=vbitrate,
            movflags=movflags,
            **video_opts,
        )
    else:
        # HLS format with fMP4 segments
        # Both MP4 and HLS streaming modes use this - difference is in serving:
        # - mp4: Stream concatenates init + segments for progressive playback
        # - hls: Browser requests segments via .m3u8 playlist
        if all_audio_tracks:
            # One video rendition plus an audio rendition per track, grouped so the
            # player can swap audio mid-song. output_file becomes the master
            # playlist, and %v names each rendition's files (v, a0, a1, ...).
            renditions = ["v:0,agroup:aud,name:v"] + [
                f"a:{i},agroup:aud,name:a{i}" + (",default:yes" if i == audio_track else "")
                for i in track_indexes
            ]
            streams = [video, *audios]
            playlist = fr.variant_playlist
            layout = {
                "master_pl_name": os.path.basename(fr.output_file),
                "var_stream_map": " ".join(renditions),
                "hls_fmp4_init_filename": fr.variant_init_filename,
                "hls_segment_filename": fr.variant_segment_pattern,
            }
        else:
            streams = [audios[0], video]
            playlist = fr.output_file
            layout = {
                "hls_fmp4_init_filename": fr.init_filename,
                "hls_segment_filename": fr.segment_pattern,
            }
        output = ffmpeg.output(
            *streams,
            playlist,
            vcodec=vcodec,
            acodec="aac",
            audio_bitrate="192k",
            ac=2,  # Force stereo
            ar=48000,  # Standard sample rate
            preset="ultrafast",
            f="hls",
            hls_time=3,
            hls_list_size=0,
            hls_playlist_type="event",
            hls_segment_type="fmp4",
            video_bitrate=vbitrate,
            **layout,
            **video_opts,
            **{
                "fps_mode": "cfr",
                "avoid_negative_ts": "make_zero",
            },
        )

    args = output.get_args()
    logging.debug(f"COMMAND: ffmpeg " + " ".join(args))
    return output


def get_ffmpeg_version() -> str:
    """Get the installed FFmpeg version string.

    Returns:
        Version string, or an error message if FFmpeg is not installed
        or version cannot be parsed.
    """
    try:
        # Execute the command 'ffmpeg -version'
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Parse the first line to get the version
        first_line = result.stdout.split("\n")[0]
        version_info = first_line.split(" ")[2]  # Assumes the version info is the third element
        return version_info
    except FileNotFoundError:
        return "FFmpeg is not installed"
    except IndexError:
        return "Unable to parse FFmpeg version"


def is_transpose_enabled() -> bool:
    """Check if FFmpeg has the rubberband filter for pitch shifting.

    Returns:
        True if rubberband filter is available, False otherwise.
    """
    try:
        filters = subprocess.run(["ffmpeg", "-filters"], capture_output=True)
    except FileNotFoundError:
        return False
    except IndexError:
        return False
    return "rubberband" in filters.stdout.decode()


def supports_hardware_h264_encoding() -> bool:
    """Check if hardware H.264 encoding (h264_v4l2m2m) is available.

    Only returns True on ARM architecture (Raspberry Pi) where h264_v4l2m2m
    is actually supported. On x86/Intel systems, returns False to use software encoding.

    Returns:
        True if hardware encoding is available, False otherwise.
    """
    # Check CPU architecture first - h264_v4l2m2m only works on ARM
    arch = platform.machine().lower()
    is_arm = any(arm_variant in arch for arm_variant in ["arm", "aarch"])

    if not is_arm:
        # Not ARM (probably Intel x86/x64), don't use h264_v4l2m2m
        logging.debug(f"CPU architecture {arch} is not ARM, using software encoder")
        return False

    if is_running_in_docker():
        # Docker containers do not have access to the GPU
        logging.debug("Running in Docker where GPU access is not available, using software encoder")
        return False

    # On ARM, check if h264_v4l2m2m is available
    try:
        codecs = subprocess.run(["ffmpeg", "-codecs"], capture_output=True)
    except FileNotFoundError:
        return False
    except IndexError:
        return False

    has_encoder = "h264_v4l2m2m" in codecs.stdout.decode()
    if has_encoder:
        logging.info("ARM platform detected, using h264_v4l2m2m hardware encoder")
    else:
        logging.debug("ARM platform but h264_v4l2m2m not available")

    return has_encoder


def is_ffmpeg_installed() -> bool:
    """Check if FFmpeg is installed and accessible.

    Returns:
        True if FFmpeg is installed, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    except FileNotFoundError:
        return False
    return True
