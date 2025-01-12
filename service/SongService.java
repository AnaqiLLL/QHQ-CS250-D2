package com.gk.study.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.gk.study.entity.Song;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

public interface SongService extends IService<Song> {
    Song findByName(String name);

    List<Song> listByEmotion(String emotion);

    void saveSongs(MultipartFile[] files) throws IOException;
}
