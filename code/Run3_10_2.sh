for nhu in 600
do
	for nhl in 1 2 3 4 5
	do
		echo sudo python code/DBN_LS_Alr2_Acc_pre_4.py AllData_MT_2.pkl.gz 0.1 $nhl $nhu 2 100 300 B2Out.txt
		echo '28812001' | sudo -kS python code/DBN_LS_Alr2_Acc_pre_xx_4.py AllData_MT_2.pkl.gz 0.1 $nhl $nhu 2 100 200 552 B10-2-2.txt
	done
done
