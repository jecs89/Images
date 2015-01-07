QT += widgets
qtHaveModule(printsupport): QT += printsupport

CONFIG += c++11

HEADERS       = imageviewer.h
SOURCES       = imageviewer.cpp \
                main.cpp
LIBS += `pkg-config opencv --libs`  -L/usr/local/cuda-6.5/lib64/

# install
target.path = $$[QT_INSTALL_EXAMPLES]/widgets/widgets/imageviewer
INSTALLS += target


wince*: {
   DEPLOYMENT_PLUGIN += qjpeg qgif
}
