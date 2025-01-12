package com.gk.study.spotify;

/**
 * @author zhangqingqing
 * @date 2025/1/4
 */
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.classic.methods.HttpPost;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.StringEntity;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;


public class SpotifyRecommendations {
    private static final String CLIENT_ID = "be1be464db194e21b33a752677e1d423"; // 替换为您的客户端 ID
    private static final String CLIENT_SECRET = "fdf667f6bd144720bed28bfe98996618"; // 替换为您的客户端密钥
    private static final String TOKEN_URL = "https://accounts.spotify.com/api/token";
    private static final String RECOMMENDATIONS_URL = "https://api.spotify.com/v1/users/smedjan/playlists";

    // 获取访问令牌
    public static String getAccessToken() throws IOException, ParseException {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost post = new HttpPost(TOKEN_URL);
            String authHeader = Base64.getEncoder().encodeToString((CLIENT_ID + ":" + CLIENT_SECRET).getBytes(StandardCharsets.UTF_8));
            post.setHeader("Authorization", "Basic " + authHeader);
            post.setHeader("Content-Type", "application/x-www-form-urlencoded");
            post.setEntity(new StringEntity("grant_type=client_credentials"));

            try (CloseableHttpResponse response = httpClient.execute(post)) {
                String responseBody = new String(response.getEntity().getContent().readAllBytes(), StandardCharsets.UTF_8);
                JsonObject jsonObject = JsonParser.parseString(responseBody).getAsJsonObject();
                return jsonObject.get("access_token").getAsString();
            }
        }
    }

    // 获取推荐歌曲
    private static List<String> getRecommendations(String accessToken) throws IOException, ParseException {
        List<String> songList = new ArrayList<>();

        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            // 调整查询参数
            String queryParams = "?seed_artists=4NHQUGzhtTLFvgF5SZesLK&seed_genres=pop&seed_tracks=0c6xIDDpzE81m2q797ordA";
            HttpGet get = new HttpGet(RECOMMENDATIONS_URL + queryParams);
            get.setHeader("Authorization", "Bearer " + accessToken);

            try (CloseableHttpResponse response = httpClient.execute(get)) {
                int statusCode = response.getCode();
                String responseBody = new String(response.getEntity().getContent().readAllBytes(), StandardCharsets.UTF_8);

                // 打印响应状态码和内容
                System.out.println("Response Code: " + statusCode);
                System.out.println("Response Body: " + responseBody);

                // 检查是否成功
                if (statusCode == 200) {
                    JsonObject jsonObject = JsonParser.parseString(responseBody).getAsJsonObject();
                    JsonArray tracks = jsonObject.getAsJsonArray("tracks");

                    for (int i = 0; i < tracks.size(); i++) {
                        JsonObject track = tracks.get(i).getAsJsonObject();
                        String songName = track.get("name").getAsString();
                        String artistName = track.getAsJsonArray("artists").get(0).getAsJsonObject().get("name").getAsString();
                        songList.add(songName + " by " + artistName);
                    }
                } else {
                    System.err.println("Error: Received status code " + statusCode);
                }
            }
        }
        return songList;
    }


    public static void main(String[] args) {
        try {
            String accessToken = getAccessToken();
            getRecommendations(accessToken);
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }
}

