#include "image.h"

#include <iostream>
#include <string>
#include <wand/MagickWand.h>

using namespace std;

void cannot_load_image(const string &filename)
{
	cerr << "Cannot load image: '" << filename << "'" << endl;
	exit(-1);
}

ub_image load_ub_image(const string &filename)
{
	MagickWandGenesis();
	MagickWand *img = NewMagickWand();
	int status = MagickReadImage(img, filename.c_str());
	ub_image loaded;
	if (status == MagickFalse) 
		cannot_load_image(filename);
	MagickFlipImage(img);
	loaded.w = MagickGetImageWidth(img);
	loaded.h = MagickGetImageHeight(img);
	loaded.data = ((unsigned char*)malloc(loaded.w * loaded.h * 4));
	unsigned char *tmp = ((unsigned char*)malloc(loaded.w * loaded.h * 4));
	int pixels = loaded.w * loaded.h;
	MagickExportImagePixels(img, 0, 0, loaded.w, loaded.h, "RGB", CharPixel, ((void*)tmp));
	for (int i = 0; i < pixels; ++i) {
 		loaded.data[4*i+0] = tmp[3*i+0];
 		loaded.data[4*i+1] = tmp[3*i+1];
 		loaded.data[4*i+2] = tmp[3*i+2];
	}
	free(tmp);
	DestroyMagickWand(img);
	MagickWandTerminus();
	return loaded;
}

void store_ub_image(const ub_image &image, const string &filename)
{
	cout << "storing to " << filename << endl;
	MagickWandGenesis();
	MagickWand *img = NewMagickWand();
	PixelWand *color = NewPixelWand();
	PixelSetColor(color, "white");
	MagickNewImage(img, image.w, image.h, color);
	unsigned char *tmp = ((unsigned char*)malloc(image.w * image.h * 3));
	int pixels = image.w * image.h;
	for (int i = 0; i < pixels; ++i) {
 		tmp[3*i+0] = image.data[0+4*i];
 		tmp[3*i+1] = image.data[1+4*i];
 		tmp[3*i+2] = image.data[2+4*i];
	}
	MagickImportImagePixels(img, 0, 0, image.w, image.h, "RGB", CharPixel, ((void*)tmp));
	free(tmp);
	MagickFlipImage(img);
	MagickWriteImage(img, filename.c_str());
	DestroyPixelWand(color);
	DestroyMagickWand(img);
	MagickWandTerminus();
}




/* vim: set foldmethod=marker: */

