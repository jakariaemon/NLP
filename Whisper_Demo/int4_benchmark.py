import os
import time
import psutil
import GPUtil
import onnx
import numpy as np
import time
from transformers import WhisperProcessor, AutoConfig, LogitsProcessorList, PretrainedConfig
import torch
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq  
from pydub import AudioSegment

file_path = "testjpcut30s.wav" 
def audiosegment_to_librosawav(audiosegment):
    # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
    channel_sounds = audiosegment.split_to_mono()[:1]   
    samples = [s.get_array_of_samples() for s in channel_sounds]
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)
    return fp_arr

def monitor_resources():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  
    cpu_usage = psutil.cpu_percent(interval=1)
    GPUs = GPUtil.getGPUs()
    gpu_usage = 0
    for gpu in GPUs:
        gpu_usage += gpu.load  
    return cpu_usage, ram_usage, gpu_usage

model_name = 'openai/whisper-large-v2'
onnx_model_path = 'whisper-large-v2-onnx-int4'
config = PretrainedConfig.from_pretrained(model_name)
sess_options = ort.SessionOptions()
#sess_options.intra_op_num_threads = args.cores_per_instance
sessions = ORTModelForSpeechSeq2Seq.load_model(
        os.path.join(onnx_model_path , 'encoder_model.onnx'),
        os.path.join(onnx_model_path , 'decoder_model.onnx'),
        os.path.join(onnx_model_path , 'decoder_with_past_model.onnx'),
        session_options=sess_options)
model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], config, onnx_model_path, sessions[2])
processor = WhisperProcessor.from_pretrained(model_name)
waveform = AudioSegment.from_file(file_path).set_frame_rate(16000)
waveform = audiosegment_to_librosawav(waveform)
cpu_before, ram_before, gpu_before = monitor_resources()
start_time = time.time()
input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
end_time = time.time()
cpu_after, ram_after, gpu_after = monitor_resources()
print(transcription)
cpu_used = cpu_after - cpu_before
ram_used = ram_after - ram_before
gpu_used = gpu_after - gpu_before
execution_time = end_time - start_time
print(f"Transcription: {transcription}")
print(f"Execution time: {execution_time} seconds")
print(f"CPU used: {cpu_used}%")
print(f"RAM used: {ram_used} GB")
print(f"GPU used: {gpu_used * 100}%") 
