package com.gk.study.spotify;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class LocalSongScanner {
    private static final String MUSIC_DIRECTORY = "music"; // 替换为实际路径

    public List<File> getLocalSongs() {
        File dir = new File(MUSIC_DIRECTORY);
        List<File> songs = new ArrayList<>();

        String[] audioExtensions = {".mp3", ".wav", ".flac", ".aac", ".ogg"};

        if (dir.isDirectory()) {
            for (File file : dir.listFiles()) {
                if (file.isFile() && isAudioFile(file, audioExtensions)) {
                    songs.add(file);
                }
            }
        }
        return songs;
    }

    private boolean isAudioFile(File file, String[] extensions) {
        for (String ext : extensions) {
            if (file.getName().toLowerCase().endsWith(ext)) {
                return true;
            }
        }
        return false;
    }
}