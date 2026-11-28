# reference: kaldi fbank (kaldi-native-fbank), per-window CMN, onnxruntime embedding
import sys, numpy as np, soundfile as sf, kaldi_native_fbank as knf, onnxruntime as ort
wav, out = sys.argv[1], sys.argv[2]
x, sr = sf.read(wav, dtype='float32')
if x.ndim > 1: x = x.mean(1)
assert sr == 16000
x = x[:16000*6]
o = knf.FbankOptions(); o.frame_opts.dither = 0; o.frame_opts.samp_freq = 16000; o.frame_opts.frame_length_ms=25; o.frame_opts.frame_shift_ms=10
o.mel_opts.num_bins = 80; o.energy_floor = 0.0
fb = knf.OnlineFbank(o); fb.accept_waveform(16000, (x*32768).tolist()); fb.input_finished()
f = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)
x.tofile(out+'/clip.f32'); f.tofile(out+'/fbank.f32')
s = ort.InferenceSession(sys.argv[3], providers=['CPUExecutionProvider'])
embs=[]
for a,b in [(0,150),(75,225),(0,f.shape[0]),(300,340)]:
    w = f[a:b]; w = w - w.mean(0, keepdims=True)
    embs.append(s.run(None, {'feats': w[None]})[0][0])
np.stack(embs).astype(np.float32).tofile(out+'/embs.f32')
print(f.shape, f[0,:4], embs[0][:4], np.linalg.norm(embs[0]))
