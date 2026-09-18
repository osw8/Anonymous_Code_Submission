#!/bin/bash



datasets=bnci2014001,bnci2015001,lee2019,stieger2021

[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet

[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_sppddsbn,cnnnet_dsmbn
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=tsmnet_sppddsbn,cnnnet_dsmbn


[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021_last

[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet

[ $? -eq 0 ] && datasets=bnci2014001,bnci2015001,lee2019,stieger2021_last
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet



datasets=hinss2021
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-session dataset=$datasets nnet=eegnet,shconvnet

[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject+uda dataset=$datasets nnet=tsmnet_spddsmbn,eegnet_dann,shconvnet_dann
[ $? -eq 0 ] && python main.py --multirun evaluation=inter-subject dataset=$datasets nnet=eegnet,shconvnet