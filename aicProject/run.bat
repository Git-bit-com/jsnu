@echo off
chcp 65001 >nul

echo ===============================================
echo   图书推荐系统
echo ===============================================
echo.

echo [1/5] 数据预处理...
python -m src.preprocess
if errorlevel 1 goto err

echo [2/5] 训练模型...
python -m src.train
if errorlevel 1 goto err

echo [3/5] 搜索最优融合权重 (tune)...
python -m src.tune --step 0.1
if errorlevel 1 goto err

echo [4/5] 生成全用户推荐结果...
python -m src.predict
if errorlevel 1 goto err

echo [5/5] 离线评估 F1...
python -m src.evaluate
if errorlevel 1 goto err

echo.
echo ✅ 全流程完成，按任意键退出...
pause >nul
goto end

:err
echo.
echo ❌ 中途有步骤失败，请根据上面的报错信息排查。
echo 按任意键退出...
pause >nul

:end
