package com.gk.study.controller;

import com.gk.study.entity.Song;
import com.gk.study.service.SongService;
import com.gk.study.spotify.DeleteSongsDTO;
import com.gk.study.spotify.LocalSongScanner;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
public class SongController {

    private final LocalSongScanner localSongScanner = new LocalSongScanner();

    @Autowired
    private SongService songService;

    @GetMapping("/songs/list")
    public List<File> getLocalSongs() {
        return localSongScanner.getLocalSongs();
    }

    @GetMapping("/songs/mixed")
    public ResponseEntity<MultiValueMap<String, Object>> playSongWithMetadata(@RequestParam String name) throws IOException {
        Song song = songService.findByName(name);
        if (song == null) {
            return ResponseEntity.notFound().build();
        }

        File file = new File(song.getPath());
        if (!file.exists()) {
            return ResponseEntity.notFound().build();
        }

        // 构建文件流
        FileInputStream inputStream = new FileInputStream(file);
        InputStreamResource resource = new InputStreamResource(inputStream);

        // 构建元信息
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("name", song.getName());
        metadata.put("size", song.getSize());
        metadata.put("uploadTime", song.getUploadTime());

        // 构建 Multipart 返回
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("metadata", metadata);
        body.add("file", resource);

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "inline;filename=" + file.getName())
                .body(body);
    }

    @GetMapping("/songs/play")
    public ResponseEntity<?> playSongWithUrl(@RequestParam String name) {
        Song song = songService.findByName(name);
        if (song == null) {
            return ResponseEntity.notFound().build();
        }

        File file = new File(song.getPath());
        if (!file.exists()) {
            return ResponseEntity.notFound().build();
        }

        try {

            // 返回文件资源
            Resource resource = new FileSystemResource(file);
            return ResponseEntity.ok()
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline;filename=" + file.getName())
                    .contentType(MediaType.parseMediaType("audio/mpeg"))
                    .body(resource);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).build();
        }

    }

    @GetMapping("/songs/emotion")
    public List<Song> getLocalSongs(@RequestParam String emotion) {
        return songService.listByEmotion(emotion);
    }

    @PostMapping("/songs/delete")
    public ResponseEntity deleteSongs(@RequestBody DeleteSongsDTO deleteSongsDTO) {
        // 指定文件夹路径
        String folderPath = "music";
        // 创建文件对象
        File folder = new File(folderPath);
        String songNameToDelete = deleteSongsDTO.getName();
        // 检查文件夹是否存在
        if (!folder.exists() || !folder.isDirectory()) {
            System.out.println("指定的路径不存在或不是一个文件夹。");
            return ResponseEntity.notFound().build();
        }

        // 遍历文件夹下的文件
        File[] files = folder.listFiles();
        if (files == null) {
            System.out.println("文件夹为空或无法访问。");
            return ResponseEntity.notFound().build();
        }

        for (File file : files) {
            if (file.isFile()) {
                // 去除文件名中的前后空格
                String fileName = file.getName().trim();
                // 标准化目标名称：去除空格、忽略大小写
                String normalizedSongName = songNameToDelete.trim();

                // 调试输出，帮助检查是否有差异
                System.out.println("文件名: '" + fileName + "' vs 目标名: '" + normalizedSongName + "'");
                normalizedSongName = normalizedSongName + ".mp3";
                // 判断是否包含目标名称（忽略大小写）
                if (fileName.equalsIgnoreCase(normalizedSongName)) {
                    // 删除文件
                    if (file.delete()) {
                        System.out.println("成功删除文件：" + fileName);
                    } else {
                        System.out.println("无法删除文件：" + fileName);
                    }
                }
            }
        }
        return ResponseEntity.ok().build();
    }

    @PostMapping("/upload")
    public String uploadSongs(@RequestParam("files") MultipartFile[] files) {
        try {
            songService.saveSongs(files);
            return "歌曲上传成功！";
        } catch (IOException e) {
            e.printStackTrace();
            return "歌曲上传失败：" + e.getMessage();
        }
    }

}
