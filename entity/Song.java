package com.gk.study.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("songs")
public class Song {
    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;   // 歌曲名称
    private String path;   // 歌曲路径
    private Long size;     // 文件大小
    private String emotion;   // 情绪
    private LocalDateTime uploadTime; // 上传时间
}
