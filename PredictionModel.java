package com.yourcompany;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
//json.simple.parser: 分析Json格式的資料
import org.json.simple.parser.JSONParser;
//RandomForest: 這次要用的模型-隨機森林
import weka.classifiers.trees.RandomForest;
//CVParameterSelection: 是由Weka提供的一個方法，用於在交叉驗證過程中自動選擇最佳參數。
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Evaluation;

public class PredictionModel {

    public static void main(String[] args) {
        try {
            // 讀取JSON資料
            JSONParser parser = new JSONParser();
            JSONArray dataArray = (JSONArray) parser.parse(new FileReader("C:\\Users\\alexl\\Desktop\\ML Project\\deeplearning4j-example\\src\\main\\java\\com\\yourcompany\\飽足評分訓練資料.json"));

            // 定義特徵，用attributes
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("美食"));
            ArrayList<String> genderValues = new ArrayList<>();
            genderValues.add("男");
            genderValues.add("女");
            attributes.add(new Attribute("性別", genderValues));
            attributes.add(new Attribute("年齡"));
            attributes.add(new Attribute("LBM"));
            attributes.add(new Attribute("飽足評分"));

            // 建立Instances物件
            Instances dataset = new Instances("飽足評分資料集", attributes, dataArray.size());
            dataset.setClassIndex(attributes.size() - 1);

            // 將JSON資料集轉換為Instances
            for (Object obj : dataArray) {
                JSONObject jsonObject = (JSONObject) obj;
                double[] instanceValue = new double[dataset.numAttributes()];
                instanceValue[0] = ((Long) jsonObject.get("美食")).doubleValue();
                instanceValue[1] = ((Long) jsonObject.get("性別")).doubleValue(); 
                instanceValue[2] = ((Long) jsonObject.get("年齡")).doubleValue();
                instanceValue[3] = ((Number) jsonObject.get("LBM")).doubleValue();
                instanceValue[4] = ((Number) jsonObject.get("飽足評分")).doubleValue();
                dataset.add(new DenseInstance(1.0, instanceValue));
            }

            // 超參數調校
            CVParameterSelection cvParamSel = new CVParameterSelection();
            cvParamSel.setClassifier(new RandomForest());
            cvParamSel.addCVParameter("I 100 200 5"); // 調整決策樹的數量，範圍為50到200，每次增量為50
            cvParamSel.addCVParameter("depth 10 30 5"); // 決策樹深度範圍，範圍為10到30，每次增量為10
            cvParamSel.setNumFolds(5); // 5-fold cross-validation
            cvParamSel.buildClassifier(dataset);

            // 獲取最佳參數
            String[] bestOptions = cvParamSel.getBestClassifierOptions();
            System.out.println("最佳參數: ");
            for (String option : bestOptions) {
                System.out.print(option + " ");
            }
            System.out.println();

            // 使用最佳參數重新訓練隨機森林模型
            RandomForest randomForest = new RandomForest();
            randomForest.setOptions(bestOptions);
            randomForest.buildClassifier(dataset);

            // 評估模型效能
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(randomForest, dataset, 5, new java.util.Random(1)); // 5折交叉驗證

            System.out.println(eval.toSummaryString("\n結果\n======\n", false));

            // 問卷
            Scanner scn = new Scanner(System.in);
            String again;

            do {
                // Q1
                System.out.println("Q1. 你想要預測的美食為？");
                System.out.println("a. 烤蔬菜番茄筆尖麵");
                System.out.println("b. 鐵觀音黑岩泡芙");
                System.out.println("c. 三重起司貝果");
                System.out.println("d. 日式豬排佐咖喱歐姆蛋燴飯");
                System.out.println("e. 打拋風植蔬餐盒");
                char foodChoice = scn.next().charAt(0);
                int foodValue = foodChoice - 'a';
                
                if (foodValue != 0) {
                    System.out.println("樣本數不足，無法預測");
                    return;
                }

                System.out.println("-------------------------");

                // Q2
                System.out.println("Q2. 你的生理性別為？");
                System.out.println("a. 男性");
                System.out.println("b. 女性");
                char genderChoice = scn.next().charAt(0);
                int genderValue = genderChoice - 'a';

                System.out.println("-------------------------");

                // Q3
                System.out.print("Q3. 你的年齡為：");
                int age = scn.nextInt();

                System.out.println("-------------------------");

                // Q4
                System.out.print("Q4. 你的身高為(cm)：");
                float height = scn.nextFloat();

                System.out.println("-------------------------");

                // Q5
                System.out.print("Q5. 你的體重為(kg)：");
                float weight = scn.nextFloat();

                System.out.println("-------------------------");

                // 計算LBM
                double lbm;
                if (age <= 14) {
                    lbm = 0.0215 * Math.pow(weight, 0.6469) * Math.pow(height, 0.7236) * 3.8;
                } else {
                    if (genderValue == 0) {
                        lbm = 0.407 * weight + 0.267 * height - 19.2;
                    } else {
                        lbm = 0.252 * weight + 0.473 * height - 48.3;
                    }
                }

                // 要預測的單筆數據
                double[] instanceValueToPredict = new double[dataset.numAttributes()];
                instanceValueToPredict[0] = foodValue; // 美食
                instanceValueToPredict[1] = genderValue; // 性別
                instanceValueToPredict[2] = age; // 年齡
                instanceValueToPredict[3] = lbm; // LBM
                instanceValueToPredict[4] = 0; // 飽足評分 (暫時設為0，因為這是要預測的標籤)

                Instance instanceToPredict = new DenseInstance(1.0, instanceValueToPredict);
                instanceToPredict.setDataset(dataset);

                // 進行預測
                double prediction = randomForest.classifyInstance(instanceToPredict);

                // 印出結果
                System.out.println("選擇的美食: " + getFoodName(foodValue));
                System.out.println("您的性別: " + (genderValue == 0 ? "男性" : "女性"));
                System.out.println("您的年齡: " + age);
                System.out.printf("您的瘦體重(LBM): %.1f\n", lbm);

                System.out.println("-------------------------");

                System.out.printf("預測飽足評分: %.1f\n", prediction);

                System.out.println("-------------------------");

                System.out.println("要再測一次嗎？(y/n)");
                again = scn.next();

                System.out.println("-------------------------");
            } while (again.equals("y"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String getFoodName(int foodValue) {
        switch (foodValue) {
            case 0: return "烤蔬菜番茄筆尖麵";
            case 1: return "鐵觀音黑岩泡芙";
            case 2: return "三重起司貝果";
            case 3: return "日式豬排佐咖喱歐姆蛋燴飯";
            case 4: return "打拋風植蔬餐盒";
            default: return "未知";
        }
    }
}
