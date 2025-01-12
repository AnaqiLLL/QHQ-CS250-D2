package com.gk.study.spotify;

/**
 * @author zhangqingqing
 * @date 2025/1/4
 */
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpClient;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Service
public class SpotifySearchService {

    private static final String SPOTIFY_API_URL = "https://api.spotify.com/v1/search";

    public static String searchSong(String keyword, String accessToken) throws IOException, InterruptedException {
        // URL 编码关键词，确保特殊字符（例如空格、冒号等）被正确编码
        String encodedKeyword = URLEncoder.encode(keyword, StandardCharsets.UTF_8);

        // 构建查询字符串
        String query = "q=" + encodedKeyword + "&type=track&limit=5";  // 查询前5个匹配的歌曲
        URI uri = URI.create(SPOTIFY_API_URL + "?" + query);

        // 构建 HTTP 请求
        HttpRequest request = HttpRequest.newBuilder()
                .uri(uri)
                .header("Authorization", "Bearer " + accessToken)
                .build();

        // 发送请求并返回响应
        HttpClient client = HttpClient.newHttpClient();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        // 返回搜索结果的 JSON 数据
        return response.body();
    }
}

