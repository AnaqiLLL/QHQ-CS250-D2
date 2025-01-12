package com.gk.study.properties;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * @author zhangqingqing
 * @date 2025/1/3
 */
@Data
@Configuration
@ConfigurationProperties(prefix = "upload.path")
public class File {

    private final String uploadPath = "D:/upload/";
}
