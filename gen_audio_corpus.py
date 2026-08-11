
python
#!/usr/bin/env python3
"""
gen_audio_corpus.py

Hostile audio corpus generator for robustness/fuzz testing of on-device
audio decoders and speech recognition front-ends. STDLIB ONLY.

Runs on Windows with Python 3.11. No numpy/scipy/pydub.

Usage:
    python gen_audio_corpus.py [output_dir]

Default output_dir: ./corpus
"""

import os
import sys
import math
import struct
import random
import array

random.seed(1337)

MAX_FILE_BYTES = 2 * 1024 * 1024  # 2MB cap per spec

# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def clamp16(x):
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return int(x)

def pcm16_from_floats(samples):
    return array.array('h', (clamp16(int(round(s * 32767.0))) for s in samples))

def write_bytes(path, data):
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES]
    with open(path, 'wb') as f:
        f.write(data)

def le16(v):
    return struct.pack('<H', v & 0xFFFF)

def le32(v):
    return struct.pack('<I', v & 0xFFFFFFFF)

def le32s(v):
    # signed 32, allow negative wraparound
    return struct.pack('<i', v)

def be16(v):
    return struct.pack('>H', v & 0xFFFF)

def be32(v):
    return struct.pack('>I', v & 0xFFFFFFFF)


# --------------------------------------------------------------------------
# WAV construction (hand-rolled, allows deliberately bad headers)
# --------------------------------------------------------------------------

def make_wav(
    samples_bytes,
    num_channels=1,
    sample_rate=16000,
    bits_per_sample=16,
    audio_format=1,          # 1 = PCM, 3 = IEEE float
    override_riff_size=None,
    override_data_size=None,
    override_byte_rate=None,
    override_block_align=None,
    truncate_to=None,
    corrupt_fmt_chunk_size=None,
    omit_data_chunk=False,
):
    """
    Build a WAV file byte-for-byte, allowing lies in any header field.
    """
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    fmt_chunk_size = 16
    if audio_format != 1:
        fmt_chunk_size = 18  # extended fmt for non-PCM (cbSize=0)

    fmt_body = struct.pack(
        '<HHIIHH',
        audio_format,
        num_channels,
        sample_rate,
        override_byte_rate if override_byte_rate is not None else byte_rate,
        override_block_align if override_block_align is not None else block_align,
        bits_per_sample,
    )
    if fmt_chunk_size == 18:
        fmt_body += struct.pack('<H', 0)

    fmt_chunk = b'fmt ' + le32(
        corrupt_fmt_chunk_size if corrupt_fmt_chunk_size is not None else len(fmt_body)
    ) + fmt_body

    data_size = len(samples_bytes)
    if override_data_size is not None:
        data_size_field = override_data_size
    else:
        data_size_field = data_size

    if omit_data_chunk:
        data_chunk = b''
    else:
        data_chunk = b'data' + le32(data_size_field) + samples_bytes

    riff_body = b'WAVE' + fmt_chunk + data_chunk
    riff_size = override_riff_size if override_riff_size is not None else len(riff_body)

    out = b'RIFF' + le32(riff_size) + riff_body

    if truncate_to is not None:
        out = out[:truncate_to]

    return out


def sine_samples(freq, duration_s, sample_rate, amplitude=0.8, phase=0.0):
    n = int(duration_s * sample_rate)
    out = []
    for i in range(n):
        t = i / sample_rate
        out.append(amplitude * math.sin(2 * math.pi * freq * t + phase))
    return out


def silence_samples(duration_s, sample_rate):
    n = int(duration_s * sample_rate)
    return [0.0] * n


# --------------------------------------------------------------------------
# CAF construction (minimal, hand-rolled, big-endian per spec)
# --------------------------------------------------------------------------

def make_caf(pcm_i16_samples, sample_rate=16000, channels=1,
             bad_desc_chunk_size=None, bad_data_chunk_size=None,
             bad_data_edit_count=None, truncate_to=None, omit_data=False):
    """
    Minimal CAF container:
      caff file header (4 magic + 2 version + 2 flags)
      'desc' chunk: sample rate (f64 BE), format id 'lpcm', flags, bytes/packet,
                    frames/packet, channels/frame, bits/channel
      'data' chunk: edit count (u32) + raw big-endian pcm bytes
    """
    header = b'caff' + struct.pack('>HH', 1, 0)

    desc_body = struct.pack(
        '>d4sIIIII',
        float(sample_rate),
        b'lpcm',
        0,          # format flags (0 = big-endian, not float)
        2 * channels,   # bytes per packet
        1,              # frames per packet
        channels,       # channels per frame
        16,             # bits per channel
    )
    desc_size = bad_desc_chunk_size if bad_desc_chunk_size is not None else len(desc_body)
    desc_chunk = b'desc' + struct.pack('>q', desc_size) + desc_body

    # Convert samples to big-endian 16-bit
    be_samples = b''.join(struct.pack('>h', clamp16(s)) for s in pcm_i16_samples)

    edit_count = bad_data_edit_count if bad_data_edit_count is not None else 0
    data_body = struct.pack('>I', edit_count) + be_samples
    data_size = bad_data_chunk_size if bad_data_chunk_size is not None else len(data_body)

    if omit_data:
        data_chunk = b''
    else:
        data_chunk = b'data' + struct.pack('>q', data_size) + data_body

    out = header + desc_chunk + data_chunk

    if truncate_to is not None:
        out = out[:truncate_to]

    return out


# --------------------------------------------------------------------------
# Minimal MP3 frame layer (hand-rolled ID3v2 + bogus MPEG frames)
# --------------------------------------------------------------------------

def make_mp3_garbage_id3(declared_size, actual_extra_bytes=200, bad_sync=False):
    """
    ID3v2 header: 'ID3' + version(2) + flags(1) + size(4, syncsafe-ish but we lie)
    followed by junk "tag" bytes, then either a valid-ish or corrupted MPEG
    frame sync word, then garbage frame bytes.
    """
    id3_header = b'ID3' + bytes([0x03, 0x00]) + bytes([0x00])
    # Syncsafe size encoding normally uses 7 bits/byte; we deliberately
    # encode an oversized / malformed value to stress the parser.
    size_bytes = struct.pack('>I', declared_size & 0x7F7F7F7F)
    id3_header += size_bytes

    junk = bytes([random.randint(0, 255) for _ in range(actual_extra_bytes)])

    if bad_sync:
        frame_sync = bytes([0x00, 0x00, 0x00, 0x00])  # invalid sync
    else:
        # 0xFFFB = MPEG1 Layer3, no CRC, 128kbps-ish bit pattern
        frame_sync = bytes([0xFF, 0xFB, 0x90, 0x64])

    frame_junk = bytes([random.randint(0, 255) for _ in range(300)])

    return id3_header + junk + frame_sync + frame_junk


# --------------------------------------------------------------------------
# ADTS AAC hand-rolled headers
# --------------------------------------------------------------------------

def make_adts_frame(sample_rate_index, channel_config, frame_length,
                     profile=1, payload_size=200, truncate_payload=False):
    """
    Build one crafted ADTS header + payload.
    ADTS fixed header (28 bits) + variable header (28 bits) simplified to
    the common 7-byte no-CRC layout.
    """
    syncword = 0xFFF
    id_bit = 0          # MPEG-4
    layer = 0
    protection_absent = 1

    profile_bits = profile & 0x3
    sr_index = sample_rate_index & 0xF
    private_bit = 0
    ch_cfg = channel_config & 0x7
    orig_copy = 0
    home = 0

    copyright_id_bit = 0
    copyright_id_start = 0
    frame_len = frame_length & 0x1FFF
    buffer_fullness = 0x7FF
    num_frames_minus_1 = 0

    b0 = (syncword >> 4) & 0xFF
    b1 = ((syncword & 0xF) << 4) | (id_bit << 3) | (layer << 1) | protection_absent
    b2 = (profile_bits << 6) | (sr_index << 2) | (private_bit << 1) | ((ch_cfg >> 2) & 0x1)
    b3 = ((ch_cfg & 0x3) << 6) | (orig_copy << 5) | (home << 4) | \
         (copyright_id_bit << 3) | (copyright_id_start << 2) | ((frame_len >> 11) & 0x3)
    b4 = (frame_len >> 3) & 0xFF
    b5 = ((frame_len & 0x7) << 5) | ((buffer_fullness >> 6) & 0x1F)
    b6 = ((buffer_fullness & 0x3F) << 2) | num_frames_minus_1

    header = bytes([b0, b1, b2, b3, b4, b5, b6])

    payload = bytes([random.randint(0, 255) for _ in range(payload_size)])
    if truncate_payload:
        payload = payload[: max(0, payload_size // 3)]

    return header + payload


# --------------------------------------------------------------------------
# Corpus writer
# --------------------------------------------------------------------------

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'corpus'
    os.makedirs(out_dir, exist_ok=True)

    manifest = []

    def add(name, data):
        path = os.path.join(out_dir, name)
        write_bytes(path, data)
        manifest.append((name, len(data)))

    # ======================================================================
    # A: Decoder attacks — WAV header lies
    # ======================================================================

    base_tone = pcm16_from_floats(sine_samples(440.0, 1.0, 16000))
    base_tone_bytes = base_tone.tobytes()

    add("A01_wav_control_valid.wav", make_wav(base_tone_bytes, sample_rate=16000))

    add("A02_wav_riff_size_huge.wav", make_wav(
        base_tone_bytes, override_riff_size=0x7FFFFFFF))

    add("A03_wav_riff_size_zero.wav", make_wav(
        base_tone_bytes, override_riff_size=0))

    add("A04_wav_data_size_huge.wav", make_wav(
        base_tone_bytes, override_data_size=0x7FFFFFFF))

    add("A05_wav_data_size_negative.wav", make_wav(
        base_tone_bytes, override_data_size=0xFFFFFFFF))

    add("A06_wav_data_size_zero_but_present.wav", make_wav(
        base_tone_bytes, override_data_size=0))

    add("A07_wav_samplerate_zero.wav", make_wav(
        base_tone_bytes, sample_rate=0))

    add("A08_wav_samplerate_max.wav", make_wav(
        base_tone_bytes, sample_rate=0xFFFFFFFF))

    add("A09_wav_samplerate_384000.wav", make_wav(
        base_tone_bytes, sample_rate=384000))

    add("A10_wav_bitspersample_zero.wav", make_wav(
        base_tone_bytes, bits_per_sample=0))

    add("A11_wav_bitspersample_weird_24.wav", make_wav(
        base_tone_bytes, bits_per_sample=24))

    add("A12_wav_bitspersample_huge_128.wav", make_wav(
        base_tone_bytes, bits_per_sample=128))

    add("A13_wav_byterate_zero.wav", make_wav(
        base_tone_bytes, override_byte_rate=0))

    add("A14_wav_byterate_huge.wav", make_wav(
        base_tone_bytes, override_byte_rate=0xFFFFFFFF))

    add("A15_wav_blockalign_zero.wav", make_wav(
        base_tone_bytes, override_block_align=0))

    add("A16_wav_blockalign_huge.wav", make_wav(
        base_tone_bytes, override_block_align=0xFFFF))

    add("A17_wav_fmt_chunk_size_lie_small.wav", make_wav(
        base_tone_bytes, corrupt_fmt_chunk_size=2))

    add("A18_wav_fmt_chunk_size_lie_huge.wav", make_wav(
        base_tone_bytes, corrupt_fmt_chunk_size=0x7FFFFFFF))

    add("A19_wav_no_data_chunk.wav", make_wav(
        base_tone_bytes, omit_data_chunk=True))

    full_wav = make_wav(base_tone_bytes, sample_rate=16000)
    add("A20_wav_truncated_header_cut.wav", full_wav[:20])
    add("A21_wav_truncated_mid_fmt.wav", full_wav[:30])
    add("A22_wav_truncated_mid_data.wav", full_wav[: len(full_wav) - 500])
    add("A23_wav_truncated_1byte.wav", full_wav[:1])
    add("A24_wav_truncated_0byte.wav", b'')

    # declared data present but file physically ends before it
    add("A25_wav_declared_data_missing.wav", make_wav(
        b'', override_data_size=len(base_tone_bytes)))

    # ======================================================================
    # A: CAF hand-rolled attacks
    # ======================================================================

    caf_tone = list(base_tone)

    add("A26_caf_control_valid.caf", make_caf(caf_tone, sample_rate=16000))

    add("A27_caf_desc_size_lie_huge.caf", make_caf(
        caf_tone, bad_desc_chunk_size=0x7FFFFFFF))

    add("A28_caf_desc_size_lie_negative.caf", make_caf(
        caf_tone, bad_desc_chunk_size=-1))

    add("A29_caf_data_size_lie_huge.caf", make_caf(
        caf_tone, bad_data_chunk_size=0x7FFFFFFF))

    add("A30_caf_data_size_lie_zero.caf", make_caf(
        caf_tone, bad_data_chunk_size=0))

    add("A31_caf_no_data_chunk.caf", make_caf(
        caf_tone, omit_data=True))

    full_caf = make_caf(caf_tone, sample_rate=16000)
    add("A32_caf_truncated_header.caf", full_caf[:10])
    add("A33_caf_truncated_mid_desc.caf", full_caf[:20])
    add("A34_caf_truncated_mid_data.caf", full_caf[: len(full_caf) - 300])

    # ======================================================================
    # A: MP3 garbage ID3 / sync attacks
    # ======================================================================

    add("A35_mp3_id3_huge_declared_size.mp3", make_mp3_garbage_id3(
        declared_size=0x7FFFFFFF, bad_sync=False))

    add("A36_mp3_id3_bad_sync_word.mp3", make_mp3_garbage_id3(
        declared_size=1000, bad_sync=True))

    add("A37_mp3_id3_zero_size.mp3", make_mp3_garbage_id3(
        declared_size=0, bad_sync=False))

    add("A38_mp3_no_id3_raw_frames.mp3",
        bytes([0xFF, 0xFB, 0x90, 0x64]) + bytes([random.randint(0, 255) for _ in range(400)]))

    add("A39_mp3_truncated_after_id3.mp3", make_mp3_garbage_id3(
        declared_size=5000, bad_sync=False)[:30])

    # ======================================================================
    # A: ADTS AAC crafted headers
    # ======================================================================

    add("A40_adts_control_valid.aac", make_adts_frame(
        sample_rate_index=3, channel_config=1, frame_length=207, payload_size=200))

    add("A41_adts_samplerate_reserved_13.aac", make_adts_frame(
        sample_rate_index=13, channel_config=1, frame_length=207))

    add("A42_adts_samplerate_reserved_14.aac", make_adts_frame(
        sample_rate_index=14, channel_config=1, frame_length=207))

    add("A43_adts_samplerate_reserved_15.aac", make_adts_frame(
        sample_rate_index=15, channel_config=1, frame_length=207))

    add("A44_adts_channelconfig_reserved_7.aac", make_adts_frame(
        sample_rate_index=3, channel_config=7, frame_length=207))

    add("A45_adts_channelconfig_reserved_8.aac", make_adts_frame(
        sample_rate_index=3, channel_config=8 & 0x7, frame_length=207))

    add("A46_adts_framelength_huge.aac", make_adts_frame(
        sample_rate_index=3, channel_config=1, frame_length=0x1FFF, payload_size=50))

    add("A47_adts_framelength_zero.aac", make_adts_frame(
        sample_rate_index=3, channel_config=1, frame_length=0, payload_size=50))

    add("A48_adts_truncated_frame.aac", make_adts_frame(
        sample_rate_index=3, channel_config=1, frame_length=207,
        payload_size=200, truncate_payload=True))

    # ======================================================================
    # B: ASR front-end attacks (valid containers, hostile content)
    # ======================================================================

    sr = 16000

    # DC offset ramps
    ramp = [max(-1.0, min(1.0, -1.0 + 2.0 * i / (sr * 2))) for i in range(sr * 2)]
    add("B01_dc_ramp.wav", make_wav(pcm16_from_floats(ramp).tobytes(), sample_rate=sr))

    dc_full = [0.99] * (sr * 2)
    add("B02_dc_offset_full_scale.wav", make_wav(pcm16_from_floats(dc_full).tobytes(), sample_rate=sr))

    # Alternating full-scale square waves at various frequencies
    for freq in [1, 10, 100, 1000, 4000, 8000]:
        n = sr * 2
        sq = []
        for i in range(n):
            t = i / sr
            cyc = t * freq
            sq.append(1.0 if (cyc - math.floor(cyc)) < 0.5 else -1.0)
        add(f"B03_square_{freq}hz.wav", make_wav(pcm16_from_floats(sq).tobytes(), sample_rate=sr))

    # Silence with periodic single-sample impulses
    imp = [0.0] * (sr * 3)
    for i in range(0, len(imp), 500):
        imp[i] = 1.0
    add("B04_impulse_every_500_samples.wav", make_wav(pcm16_from_floats(imp).tobytes(), sample_rate=sr))

    # Exponential decay tail
    decay = []
    n = sr * 2
    for i in range(n):
        t = i / sr
        decay.append(math.sin(2 * math.pi * 440 * t) * math.exp(-3.0 * t))
    add("B05_exponential_decay.wav", make_wav(pcm16_from_floats(decay).tobytes(), sample_rate=sr))

    # Amplitude clipping bursts
    clip = sine_samples(440, 1.0, sr, amplitude=3.0)  # will clamp hard
    add("B06_clipping_burst.wav", make_wav(pcm16_from_floats(clip).tobytes(), sample_rate=sr))

    # Sample rate variety
    for rate in [8000, 44100, 96000]:
        tone = sine_samples(440, 1.0, rate)
        add(f"B07_samplerate_{rate}.wav", make_wav(pcm16_from_floats(tone).tobytes(), sample_rate=rate))

    # Multi-channel WAVs
    for ch in [1, 2, 4, 6, 8]:
        n = sr
        interleaved = []
        for i in range(n):
            t = i / sr
            for c in range(ch):
                interleaved.append(0.5 * math.sin(2 * math.pi * (440 + c * 50) * t))
        add(f"B08_channels_{ch}.wav", make_wav(
            pcm16_from_floats(interleaved).tobytes(), num_channels=ch, sample_rate=sr))

    # Extremely long silent lead-in then speech-like buzz
    lead_silence = silence_samples(60.0, sr)
    speechish = []
    for i in range(sr):
        t = i / sr
        val = 0.3 * math.sin(2 * math.pi * 200 * t) + 0.2 * math.sin(2 * math.pi * 800 * t) \
              + 0.1 * math.sin(2 * math.pi * 1600 * t)
        speechish.append(val)
    long_lead = lead_silence + speechish
    add("B09_long_silence_leadin_60s.wav", make_wav(
        pcm16_from_floats(long_lead).tobytes(), sample_rate=sr))

    # Extremely short "speech" (1ms)
    short_speech = sine_samples(600, 0.001, sr)
    add("B10_speech_1ms.wav", make_wav(pcm16_from_floats(short_speech).tobytes(), sample_rate=sr))

    # Abrupt onset (VAD stress) — silence then instant full-scale tone, no ramp
    abrupt = silence_samples(0.5, sr) + sine_samples(500, 0.5, sr, amplitude=0.95)
    add("B11_abrupt_onset.wav", make_wav(pcm16_from_floats(abrupt).tobytes(), sample_rate=sr))

    # Speech-like noise: formant-ish buzz via summed sines with slow modulation
    formant_buzz = []
    n = sr * 2
    for i in range(n):
        t = i / sr
        f0 = 120 + 20 * math.sin(2 * math.pi * 2 * t)
        val = (0.4 * math.sin(2 * math.pi * f0 * t)
               + 0.25 * math.sin(2 * math.pi * (f0 * 4.2) * t)
               + 0.15 * math.sin(2 * math.pi * (f0 * 7.9) * t)
               + 0.1 * random.uniform(-1, 1))
        formant_buzz.append(max(-1.0, min(1.0, val)))
    add("B12_formant_buzz.wav", make_wav(pcm16_from_floats(formant_buzz).tobytes(), sample_rate=sr))

    # DTMF dual-tone bursts
    dtmf_pairs = [(697, 1209), (770, 1336), (852, 1477), (941, 1633)]
    dtmf_samples = []
    for (f1, f2) in dtmf_pairs:
        n = int(0.2 * sr)
        for i in range(n):
            t = i / sr
            dtmf_samples.append(0.5 * (math.sin(2 * math.pi * f1 * t) + math.sin(2 * math.pi * f2 * t)))
        dtmf_samples.extend([0.0] * int(0.05 * sr))
    add("B13_dtmf_bursts.wav", make_wav(pcm16_from_floats(dtmf_samples).tobytes(), sample_rate=sr))

    # 20kHz ultrasonic sweep (at 44.1kHz to be representable)
    sr_us = 44100
    sweep = []
    n = sr_us * 2
    f_start, f_end = 15000, 22000
    for i in range(n):
        t = i / sr_us
        frac = i / n
        f = f_start + (f_end - f_start) * frac
        sweep.append(0.6 * math.sin(2 * math.pi * f * t))
    add("B14_ultrasonic_sweep.wav", make_wav(pcm16_from_floats(sweep).tobytes(), sample_rate=sr_us))

    # Phase-inverted "speech" (buzz) pair
    inv_buzz = [-x for x in formant_buzz]
    add("B15_formant_buzz_phase_inverted.wav", make_wav(
        pcm16_from_floats(inv_buzz).tobytes(), sample_rate=sr))

    # Gap-encoded morse-like pulses
    morse = []
    pattern = [1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
    unit = int(0.15 * sr)
    for bit in pattern:
        val = 0.8 if bit else 0.0
        seg = [val * math.sin(2 * math.pi * 1000 * (i / sr)) if bit else 0.0 for i in range(unit)]
        morse.extend(seg)
    add("B16_morse_pulses.wav", make_wav(pcm16_from_floats(morse).tobytes(), sample_rate=sr))

    # Double-speed and half-speed "hello"-like phrase (synth formant glide)
    def synth_hello(duration_s, sample_rate):
        n = int(duration_s * sample_rate)
        out = []
        for i in range(n):
            t = i / sample_rate
            frac = t / duration_s
            f0 = 150 + 80 * math.sin(math.pi * frac)
            val = (0.5 * math.sin(2 * math.pi * f0 * t)
                   + 0.2 * math.sin(2 * math.pi * f0 * 2.0 * t))
            envelope = math.sin(math.pi * frac) if 0 <= frac <= 1 else 0.0
            out.append(val * max(0.0, envelope))
        return out

    hello_normal = synth_hello(0.6, sr)
    add("B17_hello_normal.wav", make_wav(pcm16_from_floats(hello_normal).tobytes(), sample_rate=sr))

    hello_double = synth_hello(0.3, sr)
    add("B18_hello_double_speed.wav", make_wav(pcm16_from_floats(hello_double).tobytes(), sample_rate=sr))

    hello_half = synth_hello(1.2, sr)
    add("B19_hello_half_speed.wav", make_wav(pcm16_from_floats(hello_half).tobytes(), sample_rate=sr))

    hello_reversed = list(reversed(hello_normal))
    add("B20_hello_reversed.wav", make_wav(pcm16_from_floats(hello_reversed).tobytes(), sample_rate=sr))

    # White noise bursts at full scale
    white = [random.uniform(-1.0, 1.0) for _ in range(sr * 2)]
    add("B21_white_noise_full_scale.wav", make_wav(pcm16_from_floats(white).tobytes(), sample_rate=sr))

    # ======================================================================
    # C: Degenerate / boundary files
    # ======================================================================

    add("C01_zero_byte_file.wav", b'')
    add("C02_one_byte_file.wav", b'\x00')

    header_only = make_wav(b'', sample_rate=sr)
    add("C03_header_only_no_samples.wav", header_only)

    declared_missing = make_wav(b'', override_data_size=44100 * 2, sample_rate=sr)
    add("C04_declared_but_missing_data.wav", declared_missing)

    # int16 extreme values
    int16_extremes = array.array('h', [32767, -32768] * (sr // 2))
    add("C05_int16_min_max_extremes.wav", make_wav(
        int16_extremes.tobytes(), sample_rate=sr))

    # int32 extreme values in a fake 32-bit PCM WAV
    int32_extremes = b''.join(
        struct.pack('<i', v) for v in ([2147483647, -2147483648] * (sr // 2))
    )
    add("C06_int32_min_max_extremes.wav", make_wav(
        int32_extremes, sample_rate=sr, bits_per_sample=32, audio_format=1))

    # IEEE float32 WAV with NaN / Inf / denormals
    float_pattern = []
    nan_val = struct.unpack('<f', struct.pack('<I', 0x7FC00000))[0]
    inf_val = struct.unpack('<f', struct.pack('<I', 0x7F800000))[0]
    neg_inf_val = struct.unpack('<f', struct.pack('<I', 0xFF800000))[0]
    denorm_val = struct.unpack('<f', struct.pack('<I', 0x00000001))[0]
    pattern_vals = [nan_val, inf_val, neg_inf_val, denorm_val, 0.0, 1.0, -1.0]
    for i in range(sr // 2):
        float_pattern.append(pattern_vals[i % len(pattern_vals)])
    float_bytes = b''.join(struct.pack('<f', v) for v in float_pattern)
    add("C07_float32_nan_inf_denormal.wav", make_wav(
        float_bytes, sample_rate=sr, bits_per_sample=32, audio_format=3))

    # Sample data exactly at boundaries mixed with normal tone
    boundary_mix = []
    tone_part = sine_samples(440, 0.5, sr)
    boundary_mix.extend(tone_part)
    boundary_mix.extend([1.0, -1.0] * 1000)
    add("C08_boundary_mix_clip_tone.wav", make_wav(
        pcm16_from_floats(boundary_mix).tobytes(), sample_rate=sr))

    # ======================================================================
    # Manifest
    # ======================================================================

    manifest_path = os.path.join(out_dir, "_manifest.txt")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f"SpeechProbe corpus manifest — {len(manifest)} files\n")
        f.write("=" * 60 + "\n")
        for name, size in manifest:
            f.write(f"{name:50s} {size:8d} bytes\n")

    print(f"Generated {len(manifest)} files into: {out_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == '__main__':
    main()


