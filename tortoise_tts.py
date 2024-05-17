# Imports used through the rest of this demo.
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython
# from audio import audioplay

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
# tts = TextToSpeech(use_deepspeed=True, kv_cache=True) # deepspeed compatibility issue
tts = TextToSpeech(kv_cache=True)

# This is the text that will be spoken.
text = "Joining two modalities results in a surprising increase in generalization! What would happen if we combined them all?"

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "ultra_fast"

ref_wav_file = 'tortoise/voices/tom/1.wav'
IPython.display.Audio(ref_wav_file)

# Pick one of the voices from the output above
voice = 'tom'

# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)

# Save output audio
output_dir = os.path.join(os.getcwd(), 'results')
assert os.path.isdir(output_dir), 'output dir: {} does not exist!'.format(output_dir)
output_file = os.path.join(output_dir, 'generated.wav')             
torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
