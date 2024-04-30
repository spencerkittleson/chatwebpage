from faster_whisper import WhisperModel

# ffmpeg -i 'video_file.mp4' -q:a 0 -map a audio.mp3
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = model.transcribe('audio.mp3', vad_filter=True)
print('get building')
segments = list(segments)
text = ''
for segment in segments:
    text = text + segment.text
print('write to file')
with open('transcribe.txt', 'w') as file:
    file.write(text)

print('done')