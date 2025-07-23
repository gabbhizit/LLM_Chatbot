"""import requests
import certifi

response = requests.get("https://en.wikipedia.org", verify=certifi.where())
print(response.status_code)"""

from youtube_transcript_api import YouTubeTranscriptApi
import youtube_transcript_api
print(youtube_transcript_api.__file__)


video_id = "vJOGC8QJZJQ"
ytt= YouTubeTranscriptApi()
try:
    transcript_list = ytt.list(video_id=video_id)
    transcript = transcript_list.find_transcript(['en'])  # or ['en', 'hi'] for fallback
    for line in transcript.fetch():
        print(line.text)
except Exception as e:
    print(f"Error: {e}")

