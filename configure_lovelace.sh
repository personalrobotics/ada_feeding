#!/bin/sh
# export LOCAL_USER=ada
ifconfig lo multicast
route add -net 224.0.0.0 netmask 240.0.0.0 dev lo
setfacl -m u:ada:rw /dev/snd/*
