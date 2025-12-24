# Numerical_PJ

使用以下命令执行正常数据集下的求解比较

python ridge_compare.py \
  --tune_lsqr \
  --lsqr_tol_list 1e-10,1e-12,1e-14,1e-16 \
  --lsqr_rmse_eps 1e-4 \
  --lsqr_check_every 20 \
  --plot \
  --save_fig normal_compare.png
  
  使用以下命令执行病态数据集下的求解比较

python ridge_compare.py --ill --ill_alpha 6 --ill_dup 8 --ill_eps 1e-6 \
  --tune_lsqr --lsqr_tol_list 1e-10,1e-12,1e-14,1e-16 --lsqr_rmse_eps 1e-4 \
  --lsqr_check_every 20 --plot --save_fig ill_compare.png
