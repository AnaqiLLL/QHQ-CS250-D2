package com.gk.study.controller;

/**
 * @author zhangqingqing
 * @date 2025/1/5
 */
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/recognize")
public class EmotionRecognitionController {

    @PostMapping("/recognize-emotion")
    public Map<String, Object> recognizeEmotion(@RequestParam("image") MultipartFile imageFile) {
        Map<String, Object> response = new HashMap<>();

        try {
            // 将图片保存到本地临时文件
            File tempFile = File.createTempFile("uploaded-", ".png");
            imageFile.transferTo(tempFile);

            // 调用 Python 脚本
            //String pythonScriptPath = "E://Part-time_Project/fer2013-master/fertestcustom.py";
            String pythonScriptPath = "E://Study_File/University/MUST/Studying/G3.1/SoftwareEngineering/FinalProject/QHQSystem/fer2013-master/fertestcustom.py";

            String result = runPythonScript(pythonScriptPath, tempFile.getAbsolutePath());

            // 返回 Python 的结果
            String trim = result.trim();
            // 定义正则表达式，匹配 "Emotion: <情绪>"
            String regex = "Emotion:\\s*(\\w+)";
            Pattern pattern = Pattern.compile(regex);
            Matcher matcher = pattern.matcher(trim);
            String emotion = "";
            // 查找匹配的情绪值
            if (matcher.find()) {
               emotion = matcher.group(1); // 返回括号中的内容
            }

            response.put("emotion", emotion);
        } catch (Exception e) {
            e.printStackTrace();
            response.put("status", "error");
            response.put("message", e.getMessage());
        }

        return response;
    }

    private String runPythonScript(String scriptPath, String imagePath) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("C:\\Users\\Anaqi\\AppData\\Local\\Programs\\Python\\Python38\\python.exe", scriptPath, imagePath);
        processBuilder.redirectErrorStream(true); // 合并标准错误和标准输出
        Process process = processBuilder.start();

        // 捕获输出
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Python script exited with error code: " + exitCode + "\nOutput:\n" + output.toString());
        }

        return output.toString();
    }

}

