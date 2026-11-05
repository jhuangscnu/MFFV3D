# fsnet ablation
bob bio hist -e fsnet-f_tex_bal_dev_scores.csv fsnet-f_tex_bal_test_scores.csv -o 'hist_bal_fsnet-f.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-f_tex_nom_dev_scores.csv fsnet-f_tex_nom_test_scores.csv -o 'hist_nom_fsnet-f.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-s_tex_bal_dev_scores.csv fsnet-s_tex_bal_test_scores.csv -o 'hist_bal_fsnet-s.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-s_tex_nom_dev_scores.csv fsnet-s_tex_nom_test_scores.csv -o 'hist_nom_fsnet-s.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-fa_tex_bal_dev_scores.csv fsnet-fa_tex_bal_test_scores.csv -o 'hist_bal_fsnet-fa.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-fa_tex_nom_dev_scores.csv fsnet-fa_tex_nom_test_scores.csv -o 'hist_nom_fsnet-fa.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-sa_tex_bal_dev_scores.csv fsnet-sa_tex_bal_test_scores.csv -o 'hist_bal_fsnet-sa.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-sa_tex_nom_dev_scores.csv fsnet-sa_tex_nom_test_scores.csv -o 'hist_nom_fsnet-sa.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-a_tex_bal_dev_scores.csv fsnet-a_tex_bal_test_scores.csv -o 'hist_bal_fsnet-a.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet-a_tex_nom_dev_scores.csv fsnet-a_tex_nom_test_scores.csv -o 'hist_nom_fsnet-a.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_tex_bal_dev_scores.csv fsnet_x_tex_bal_test_scores.csv -o 'hist_bal_fsnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_tex_nom_dev_scores.csv fsnet_x_tex_nom_test_scores.csv -o 'hist_nom_fsnet.pdf' -ts dev,test --figsize 5.5,3
# fsnet fusion
bob bio hist -e fsnet_x_sha_bal_dev_scores.csv fsnet_x_sha_bal_test_scores.csv -o 'hist_bal_fsnet_sha.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_sha_nom_dev_scores.csv fsnet_x_sha_nom_test_scores.csv -o 'hist_nom_fsnet_sha.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_fusion_bal_dev_scores.csv fsnet_x_fusion_bal_test_scores.csv -o 'hist_bal_fsnet_fusion.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_fusion_nom_dev_scores.csv fsnet_x_fusion_nom_test_scores.csv -o 'hist_nom_fsnet_fusion.pdf' -ts dev,test --figsize 5.5,3
# fsnet comparison
bob bio hist -e resnet_tex_bal_dev_scores.csv resnet_tex_bal_test_scores.csv -o 'hist_bal_resnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e resnet_tex_nom_dev_scores.csv resnet_tex_nom_test_scores.csv -o 'hist_nom_resnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e mobilenetv2_tex_bal_dev_scores.csv mobilenetv2_tex_bal_test_scores.csv -o 'hist_bal_mobilenetv2.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e mobilenetv2_tex_nom_dev_scores.csv mobilenetv2_tex_nom_test_scores.csv -o 'hist_nom_mobilenetv2.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e ghostnet_tex_bal_dev_scores.csv ghostnet_tex_bal_test_scores.csv -o 'hist_bal_ghostnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e ghostnet_tex_nom_dev_scores.csv ghostnet_tex_nom_test_scores.csv -o 'hist_nom_ghostnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e xception_tex_bal_dev_scores.csv xception_tex_bal_test_scores.csv -o 'hist_bal_xception.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e xception_tex_nom_dev_scores.csv xception_tex_nom_test_scores.csv -o 'hist_nom_xception.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e efficientnet_tex_bal_dev_scores.csv efficientnet_tex_bal_test_scores.csv -o 'hist_bal_efficientnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e efficientnet_tex_nom_dev_scores.csv efficientnet_tex_nom_test_scores.csv -o 'hist_nom_efficientnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_tex_bal_dev_scores.csv fsnet_x_tex_bal_test_scores.csv -o 'hist_bal_fsnet.pdf' -ts dev,test --figsize 5.5,3
bob bio hist -e fsnet_x_tex_nom_dev_scores.csv fsnet_x_tex_nom_test_scores.csv -o 'hist_nom_fsnet.pdf' -ts dev,test --figsize 5.5,3