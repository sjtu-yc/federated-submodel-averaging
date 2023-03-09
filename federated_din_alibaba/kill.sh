#!/bin/bash 

string=$(netstat -anp|grep 0.0.0.0:25380)
string=${string##*LISTEN}
string=${string%/*}
kill -15 $string
