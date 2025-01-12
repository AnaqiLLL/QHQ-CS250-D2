package com.gk.study.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.gk.study.entity.Song;
import com.gk.study.mapper.SongMapper;
import com.gk.study.service.SongService;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import static com.baomidou.mybatisplus.core.toolkit.Wrappers.lambdaQuery;

@Service
public class SongServiceImpl extends ServiceImpl<SongMapper, Song> implements SongService {

    @Override
    public Song findByName(String name) {
        return lambdaQuery().like(Song::getName, name).one();
    }

    @Override
    public List<Song> listByEmotion(String emotion) {
        return lambdaQuery().like(Song::getEmotion, emotion).list();
    }

    @Override
    public void saveSongs(MultipartFile[] files) throws IOException {
        // 设置临时存储路径
        String tempDir = System.getProperty("java.io.tmpdir");  // 获取系统默认的临时文件夹
        File tempFolder = new File(tempDir, "uploaded_files");
        if (!tempFolder.exists()) {
            tempFolder.mkdirs();
        }

        // 目标文件夹路径
        File targetFolder = new File("music");

        if (!targetFolder.exists()) {
            targetFolder.mkdirs();
        }

        // 遍历上传的文件
        for (MultipartFile file : files) {
            String fileName = file.getOriginalFilename();
            if (fileName == null || fileName.isEmpty()) {
                continue;
            }

            // 临时保存文件
            File tempFile = new File(tempFolder, fileName);
            file.transferTo(tempFile);

            // 将文件从临时目录移到目标目录
            File targetFile = new File(targetFolder, fileName);
            tempFile.renameTo(targetFile);  // 将文件移动到目标文件夹

            // 创建 Song 实体并保存
            Song song = new Song();
            String fileNameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));
            song.setName(fileNameWithoutExtension);
            song.setPath(targetFile.getPath().replace("\\", "/"));
            song.setSize(file.getSize());
            String[] emotions = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

            // 直接通过Random选择一个情绪并设置
            song.setEmotion(emotions[new Random().nextInt(emotions.length)]);

            save(song);
        }
    }
}
