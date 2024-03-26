from helper_functions import get_similar_songs, get_spotify_songs

playlist_link = 'https://open.spotify.com/playlist/3HNPSt5rF56ayWEt3WOQiJ?si=9c73ad1483864de8'
playlist_id = playlist_link.split("/")[-1].split("?")[0]

print(get_similar_songs(get_spotify_songs(playlist_id)))