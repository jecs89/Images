#-------------------------------------------------
#
# Project created by QtCreator 2014-09-12T18:59:07
#
#-------------------------------------------------

QT       += core

QT       -= gui

LIBS += `pkg-config opencv --cflags --libs`

TARGET = Test2
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp
