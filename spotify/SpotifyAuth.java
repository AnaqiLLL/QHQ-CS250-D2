package com.gk.study.spotify;//package com.gk.study.spotify;
//
//import com.google.gson.JsonObject;
//import com.google.gson.JsonParser;
//import org.apache.hc.client5.http.classic.methods.HttpPost;
//import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
//import org.apache.hc.client5.http.impl.classic.CloseableHttpResponse;
//import org.apache.hc.client5.http.impl.classic.HttpClients;
//import org.apache.hc.core5.http.ParseException;
//import org.apache.hc.core5.http.io.entity.StringEntity;
//
//import java.io.IOException;
//import java.nio.charset.StandardCharsets;
//import java.util.Base64;
//
///**
// * @author zhangqingqing
// * @date 2025/1/4
// */
//public class SpotifyAuth {
//
//    private static final String CLIENT_ID = "be1be464db194e21b33a752677e1d423"; // 替换为您的客户端 ID
//    private static final String CLIENT_SECRET = "fdf667f6bd144720bed28bfe98996618"; // 替换为您的客户端密钥
//    private static final String TOKEN_URL = "https://accounts.spotify.com/api/token";
//    private static final String RECOMMENDATIONS_URL = "https://api.spotify.com/v1/users/smedjan/playlists";
//
//    // 获取访问令牌
//    public static String getAccessToken() throws IOException, ParseException {
//        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
//            HttpPost post = new HttpPost(TOKEN_URL);
//            String authHeader = Base64.getEncoder().encodeToString((CLIENT_ID + ":" + CLIENT_SECRET).getBytes(StandardCharsets.UTF_8));
//            post.setHeader("Authorization", "Basic " + authHeader);
//            post.setHeader("Content-Type", "application/x-www-form-urlencoded");
//            post.setEntity(new StringEntity("grant_type=client_credentials"));
//
//            try (CloseableHttpResponse response = httpClient.execute(post)) {
//                String responseBody = new String(response.getEntity().getContent().readAllBytes(), StandardCharsets.UTF_8);
//                JsonObject jsonObject = JsonParser.parseString(responseBody).getAsJsonObject();
//                return jsonObject.get("access_token").getAsString();
//            }
//        }
//    }
//}
