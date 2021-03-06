﻿#include <stdlib.h>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <time.h>

/*COISAS KINECT*/
#include <Windows.h>
#include <Ole2.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>
#include <FaceTrackLib.h>

#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEIGHT 480
#define KINECT_DEPTH_WIDTH 640
#define KINECT_DEPTH_HEIGHT 480

IFTFaceTracker* m_pFaceTracker;
HANDLE m_hNextImageFrameEvent;
HANDLE m_hNextDepthFrameEvent;
HANDLE m_pImageStreamHandle;
HANDLE m_pDepthStreamHandle;
INuiSensor* pSensor;
INuiCoordinateMapper* pMapper;
/**********************/

/*COISAS OPENGL*/
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>

int FormWidth = 640;
int FormHeight = 480;
int mButton;
float twist, elevation, azimuth;
float cameraDistance = 0,cameraX = 0,cameraY = 0;
int xBegin, yBegin;

#define glFovy 45		
#define glZNear 1.0		
#define glZFar 150.0
/**********************/


/*COISAS FACETRACKER*/
#include <FaceTracker/Tracker.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>
char ftFile[] = "C:\\Users\\Pedro Azevedo\\Documents\\FaceTracker-master\\FaceTracker-master\\model\\face2.tracker";
char conFile[] = "C:\\Users\\Pedro Azevedo\\Documents\\FaceTracker-master\\FaceTracker-master\\model\\face.con";
char triFile[] = "C:\\Users\\Pedro Azevedo\\Documents\\FaceTracker-master\\FaceTracker-master\\model\\face.tri";
bool fcheck = false;
double scale = 1;
int fpd = -1;
bool show = true;
cv::Rect faceDim;

FACETRACKER::Tracker model(ftFile);
cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
/*******************/


using namespace cv;
using namespace std;

Mat image(KINECT_IMAGE_HEIGHT,KINECT_IMAGE_WIDTH,CV_8UC4);
Mat imageComLand(KINECT_IMAGE_HEIGHT,KINECT_IMAGE_WIDTH,CV_8UC4);
Mat imageDosLandmarks = Mat::zeros(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_8UC1);
Mat depthPure(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_16UC1);
Mat imageAcertada(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_32FC3,cv::Scalar::all(0));
Mat pointCloud_XYZ(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_32FC3,cv::Scalar::all(0));


struct pontosShape{
	float pX;
	float pY;
	float pZ;};

struct pontosDepth{
	float pX;
	float pY;
	float pZ;
	float r;
	float g;
	float b;
	float rN;
	float gN;
	float bN;};


pontosShape pShape;
pontosDepth pDepth;
struct pontosShape pontosSh[66];

//vector<pontosShape> pontosSh(66);
vector<pontosDepth> pontosDe;

cv::Mat shapeG,conG,triG,visiG;

int vet[640*480];
int vetLandMarks[640*480];

void salvaArquivoEquivalenciaLand(int frame){
		
		ofstream arq;
		stringstream ss;
		if(frame<10)
			ss << "FrameLAND_000"<<frame;
		else if(frame>9 && frame<100)
			ss << "FrameLAND_00"<<frame;
		else if(frame>99 && frame<1000)
			ss << "FrameLAND_0"<<frame;
		else if(frame>999 && frame<10000)
			ss << "FrameLAND_"<<frame;
		string fileName = ss.str();
		arq.open("D:\\MestradoUFES\\projetoMestrado\\depthAlignTrackingFace\\depthAndLandmarks\\"+fileName+".txt");
				
		//for(size_t i = 0; i<pontosSh.size();i++){
		for(int i = 0; i<66;i++){
			if (i<=26)			  arq<<pontosSh[i].pX<<" "<<pontosSh[i].pY<<" "<<pontosSh[i].pZ<<" "<<"255"<<" "<<"0"<<" "<<"0"<<endl; // Contorno facial 26
			else if (i>26&&i<=30) arq<<pontosSh[i].pX<<" "<<pontosSh[i].pY<<" "<<pontosSh[i].pZ<<" "<<"0"<<" "<<"128"<<" "<<"0"<<endl; // Nariz 4
			else if (i>30&&i<=35) arq<<pontosSh[i].pX<<" "<<pontosSh[i].pY<<" "<<pontosSh[i].pZ<<" "<<"0"<<" "<<"255"<<" "<<"0"<<endl; // Parte abaixo nariz 5
			else if (i>35&&i<=47) arq<<pontosSh[i].pX<<" "<<pontosSh[i].pY<<" "<<pontosSh[i].pZ<<" "<<"0"<<" "<<"0"<<" "<<"255"<<endl; // olhos 12
			else				  arq<<pontosSh[i].pX<<" "<<pontosSh[i].pY<<" "<<pontosSh[i].pZ<<" "<<"0"<<" "<<"0"<<" "<<"128"<<endl; // Boca 18
		}		
		arq.close();
			
}

void salvaArquivoEquivalenciaDepth(int frame){
		
		ofstream arq;
		stringstream ss;
		if(frame<10)
			ss << "FrameDEPTH_000"<<frame;
		else if(frame>9 && frame<100)
			ss << "FrameDEPTH_00"<<frame;
		else if(frame>99 && frame<1000)
			ss << "FrameDEPTH_0"<<frame;
		else if(frame>999 && frame<10000)
			ss << "FrameDEPTH_0"<<frame;
		string fileName = ss.str();
		arq.open("D:\\MestradoUFES\\projetoMestrado\\depthAlignTrackingFace\\depthAndLandmarks\\"+fileName+".txt");

		int totFaceHeight = faceDim.y - (faceDim.height);
		int totFaceWidth = faceDim.x - (faceDim.width);

		for(size_t k=0;k<pontosDe.size();k++){
			arq<<pontosDe[k].pX<<" "<<pontosDe[k].pY<<" "<<pontosDe[k].pZ<<" "<<pontosDe[k].r<<" "<<pontosDe[k].g<<" "<<pontosDe[k].b<<endl; // Contorno facial 26
		}		

		arq.close();
			
}

void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
	shapeG = shape.clone();
	conG = con.clone();
	triG = tri.clone();
	visiG = visi.clone();
		
  int i,n = shape.rows/2;
  cv::Point p1,p2;
  cv::Scalar c;

  // Desenha os 66 puntos
  for(i = 0; i < n; i++){    
    if(visi.at<int>(i,0) == 0)continue;
	//cout<<i<<"  "<<shape.at<double>(i,0)<<"    "<<shape.at<double>(i+n,0)<<endl;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
	//cout<<i<<"  "<<p1.x<<"    "<<p1.y<<endl;

	//c = CV_RGB(0,255,0);
    if (i<=26) c = CV_RGB(255,0,0);			   // Contorno facial 26
    else if (i>26&&i<=30) c = CV_RGB(0,128,0); // Nariz 4
    else if (i>30&&i<=35) c = CV_RGB(0,255,0); // Parte abaixo nariz 5
    else if (i>35&&i<=47) c = CV_RGB(0,0,255); // olhos 12
    else c = CV_RGB(0,0,128);                  // Boca 18

    cv::circle(image,p1,1,c,2);
  }

  
  return;
}

void detectaFace(Mat frame){

	std::vector<int> wSize1(1);
	std::vector<int> wSize2(3);

	int nIter = 5;
	double clamp=3,fTol=0.01;
	cv::Mat gray,im;
		
	wSize1[0] = 7;
	wSize2[0] = 11;
	wSize2[1] = 9;
	wSize2[2] = 7;
		
	bool failed = true;
	
	if(scale == 1)
		im = frame; 
	else
		cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));

	//cv::flip(im,im,1); 
	cv::cvtColor(im,gray,CV_BGR2GRAY);

	////track this image
    std::vector<int> wSize;
	if(failed)
		wSize = wSize2;
	else
		wSize = wSize1; 

	cv::Rect face;

	if(model.Track(gray,face,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
		faceDim.x = face.x;
		faceDim.y = face.y;
		faceDim.width = face.width;
		faceDim.height = face.height;
		//cout<<"DETECTEI"<<endl;
		int idx = model._clm.GetViewIdx(); 
		failed = false;
		Draw(im,model._shape,con,tri,model._clm._visi[idx]); 
	}
	else{
      if(show){
		  cv::Mat R(im,cvRect(0,0,150,50));
		  R = cv::Scalar(0,0,255);
	  }
      model.FrameReset(); failed = true;
    }

    imshow("Face Tracker",im);
	int c = cvWaitKey(10);
    if(char(c) == 'd')model.FrameReset();
	
}

//void createLandMarkImage(){
void createLandMarkImage(Mat imageDosLandmarks){

	int i;
	int n = shapeG.rows/2;
	cv::Point p1;

	for(i = 0; i < n; i++){
		
		p1 = cv::Point(shapeG.at<double>(i,0),shapeG.at<double>(i+n,0));
		imageDosLandmarks.at<uchar>(p1.y,p1.x) = 255;
	}
	//imshow("out", imageDosLandmarks);

}

void calcMinMaxValues(float &xMin, float &xMax, float &yMin, float &yMax, float &zMin, float &zMax){

	xMin = pontosSh[0].pX;
	xMax = pontosSh[0].pX;
	yMin = pontosSh[0].pY;
	yMax = pontosSh[0].pY;
	zMin = pontosSh[0].pZ;
	zMax = pontosSh[0].pZ;

	//for (size_t i=0; i<pontosSh.size(); i++){
	for (int i=0; i<66; i++){
		if(pontosSh[i].pX<xMin)
			xMin = pontosSh[i].pX;
		if(pontosSh[i].pY<yMin)
			yMin = pontosSh[i].pY;
		if(pontosSh[i].pZ<zMin)
			zMin = pontosSh[i].pZ;
		if(pontosSh[i].pX>xMax)
			xMax = pontosSh[i].pX;
		if(pontosSh[i].pY>yMax)
			yMax = pontosSh[i].pY;
		if(pontosSh[i].pZ>zMin)
			zMax = pontosSh[i].pZ;
	}

}

int GetImage(cv::Mat &image,HANDLE frameEvent,HANDLE streamHandle){
	
	const NUI_IMAGE_FRAME *pImageFrame = NULL;
	
	
	WaitForSingleObject(frameEvent,INFINITE);
	HRESULT hr = NuiImageStreamGetNextFrame(streamHandle, 30 , &pImageFrame );
	if( FAILED( hr ) ) 
		return -1;

	
	INuiFrameTexture * pTexture = pImageFrame->pFrameTexture;
	NUI_LOCKED_RECT LockedRect;
	pTexture->LockRect( 0, &LockedRect, NULL, 0 );
    
	if( LockedRect.Pitch != 0 ){

		BYTE *pBuffer = (BYTE*) LockedRect.pBits;
		memcpy(image.data,pBuffer,image.step * image.rows);
	}

	hr = NuiImageStreamReleaseFrame( streamHandle, pImageFrame );
	if( FAILED( hr ) ) 
		return -1;

	return 0;
}

void retrievePointCloudMap(Mat &depth,Mat &pointCloud_XYZ){
	 
    unsigned short* dp = (unsigned short*)depth.data; //ponteiro para os dados de profundidade
	Point3f *point = (Point3f *)pointCloud_XYZ.data; //ponteiro para os dados de profundidade em 3d
	
	for(int y = 0;y < depth.rows;y++){
		for(int x = 0;x < depth.cols;x++, dp++,point++){
			Vector4 RealPoints = NuiTransformDepthImageToSkeleton(x,y,*dp, NUI_IMAGE_RESOLUTION_640x480); //conversão dos pontos 2D p/ 3d		
			point->x = RealPoints.x/RealPoints.w;//inserindo valor de x em point
			point->y = RealPoints.y/RealPoints.w;//inserindo valor de y em point
			point->z = RealPoints.z/RealPoints.w;//inserindo valor de z em point

		}
	}	
}

void drawPointCloud(){

    glPointSize(3);
    glBegin(GL_POINTS);

	for(size_t i=0; i<pontosDe.size();i++){
		glColor3f(pontosDe[i].rN,pontosDe[i].gN,pontosDe[i].bN);
		glVertex3f(pontosDe[i].pX,pontosDe[i].pY,pontosDe[i].pZ);
	
	}

    glEnd();

	glPointSize(5);
    glBegin(GL_POINTS);

	//for(size_t i=0; i<pontosSh.size();i++){
	for(int i=0; i<66;i++){
		glColor3f(1,0,0);
		glVertex3f(pontosSh[i].pX,pontosSh[i].pY,pontosSh[i].pZ);
	
	}

    glEnd();
}

void preProcLandmarks(Mat &shapis){

	int i = 0;
	int n = shapeG.rows/2;
	cv::Point p1;

	for(int i=0; i<640*480; i++)
		vetLandMarks[i] = 0;

	for(i=0; i<n; i++){
		p1 = cv::Point(shapeG.at<double>(i,0),shapeG.at<double>(i+n,0));
		vetLandMarks[p1.y*shapis.cols+p1.x] = i; //adiciono "i" a posicao correspondente ao pixel na imagem no vetor de landmarks
	}

}

void getLandmarks3D(Mat &pointCloud_XYZ, Mat &shapis){

    static int x,y;
			
	uchar *land = (unsigned char*)(shapis.data); //ponteiro para os dados da imagem RGB
	Point3f *point = (Point3f*)pointCloud_XYZ.data; //ponteiro para os dados de profundidade XYZ
		
	LONG colorX,colorY, antX, antY; //ponto na matriz onde esta determinada cor
	int cont = 0;
	int k=0;

	for(int i=0; i<640*480; i++)
		vet[i] = 0;
	
	for(y = 0;y < pointCloud_XYZ.rows;y++){
		for(x = 0;x < pointCloud_XYZ.cols;x++,point++){					
			NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480,NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,0,&colorX,&colorY); //tem como saida a linha e coluna de cor do pixel
			if(colorX!=NULL && colorY!=NULL){
				if(0 <= colorX && colorX < shapis.cols && 0 <= colorY && colorY < shapis.rows){
					if(shapis.at<uchar>(colorY,colorX) == 255 && vet[colorY*shapis.cols+colorX]!=1){//pego os pontos brancos da imagem de landmarks para criar os landmarks 3d e verifico se o mesmo ja esta marcado
						int i = 0;
						int n = shapeG.rows/2;
						cv::Point p1;
						p1 = cv::Point(shapeG.at<double>(i,0),shapeG.at<double>(i+n,0));						
						k = vetLandMarks[colorY*shapis.cols+colorX];
						vet[colorY*shapis.cols+colorX] = 1;
						/*pShape.pX = point->x;
						pShape.pY = point->y;
						pShape.pZ = point->z;*/
						pontosSh[k].pX = point->x;
						pontosSh[k].pY = point->y;
						pontosSh[k].pZ = point->z;
						//pontosSh.push_back(pShape);
						//pontosSh.insert(pontosSh.begin()+k, pShape);
						cont++;	
						i++;
					}
				}
			}
		}
	}
	
	//cout<<k<<endl;
}

void getPointCloud3D(Mat &pointCloud_XYZ, Mat&rgbImage, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax){
		static int x,y;

    	uchar *p = (unsigned char*)(rgbImage.data); //ponteiro para os dados da imagem RGB
		Point3f *point = (Point3f*)pointCloud_XYZ.data; //ponteiro para os dados de profundidade XYZ
		
		LONG colorX,colorY; //ponto na matriz onde esta determinada cor
		float rN,gN,bN; //RGB entre 0 e 1
		float r,g,b;
		int cont = 0;

		for(int i=0; i<640*480; i++)
			vet[i] = 0;

		for(y = 0;y < pointCloud_XYZ.rows;y++){
			for(x = 0;x < pointCloud_XYZ.cols;x++,point++){				
				NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480,NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,0,&colorX,&colorY); //tem como saida a linha e coluna de cor do pixel
				
				if(0 <= colorX && colorX <= rgbImage.cols && 0 <= colorY && colorY <= rgbImage.rows){ 
					if(point->x!=0 && point->y!=0 && point->z !=0 && vet[colorY*rgbImage.cols+colorX]!=1 &&
						point->x >= xMin+(xMin/2.5) && point->x <= xMax+(xMax/2.5) &&
					   point->y >= yMin+(yMin/2.5) && point->y <= yMax+(yMax/2.5) &&
					   point->z >= zMin+(zMin/2.5) && point->z <= zMax+(zMax/2.5)){	//se estiver dentro do limite+offset e sem pontos com X,Y,Z=0, salvo no vetor					
						vet[colorY*rgbImage.cols+colorX] = 1;
						r = (p[colorY * rgbImage.step + colorX * rgbImage.channels()])  ;//pega o pixel vermelho da imagem rgb
						g = (p[colorY * rgbImage.step + colorX * rgbImage.channels()+1]);//pega o pixel verde da imagem rgb
						b = (p[colorY * rgbImage.step + colorX * rgbImage.channels()+2]);//pega o pixel azul da imagem rgb				
						rN = r/255;//converte o valor do pixel vermelho para valores entre 0 e 1
						gN = g/255;//converte o valor do pixel verde para valores entre 0 e 1
						bN = b/255;//converte o valor do pixel azul para valores entre 0 e 1
						pDepth.pX = point->x;
						pDepth.pY = point->y;
						pDepth.pZ = point->z;
						pDepth.r = r;
						pDepth.g = g;
						pDepth.b = b;
						pDepth.rN = rN;
						pDepth.gN = gN;
						pDepth.bN = bN;
						pontosDe.push_back(pDepth);
					}				
				}
			}
		}

    glEnd();	
}

void polarview(){
    glTranslatef( cameraX, cameraY, cameraDistance);
    glRotatef( -twist, 0.0, 0.0, 1.0);
    glRotatef( -elevation, 1.0, 0.0, 0.0);
    glRotatef( -azimuth, 0.0, 1.0, 0.0);
}

int pass=0;
clock_t inicio, fim;
void display(){
	inicio = clock();
    // clear screen and depth buffer
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // Reset the coordinate system before modifying
    glLoadIdentity(); 
    glEnable(GL_DEPTH_TEST); 
    gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);

	float xMin=0, xMax=0, yMin=0, yMax=0, zMin=0, zMax=0;

	if(GetImage(image,m_hNextImageFrameEvent,m_pImageStreamHandle)==-1) //retorna imagem RGB
		return;

	if(GetImage(depthPure,m_hNextDepthFrameEvent,m_pDepthStreamHandle)==-1) //retorna image de profundidade
		return;

	//mapColorFrame();

	imageComLand = image.clone();

	detectaFace(imageComLand); //detecto os landmarks da face(o que me interessa é shapeG(os pontos))

	createLandMarkImage(imageDosLandmarks); //converto shapeG para imagem
	
	//transforma os pontos de depth em uma nuvem de pontos 3D
    retrievePointCloudMap(depthPure,pointCloud_XYZ);		

	polarview();

    //imshow("depth",depthPure);
	
	//converte de BGR para RGB
    cvtColor(image,image,CV_RGBA2BGRA);

	preProcLandmarks(imageDosLandmarks); //preencho o vetLandMarks com as suas respectivas posicoes em shapeG

	getLandmarks3D(pointCloud_XYZ, imageDosLandmarks);
	//if(pontosSh.size()>0)
		calcMinMaxValues(xMin, xMax, yMin, yMax, zMin, zMax);
	//printf("\nFrame%d: %f, %f, %f, %f, %f, %f\n\n", pass,xMin, xMax, yMin, yMax, zMin, zMax);
	getPointCloud3D(pointCloud_XYZ, image, xMin, xMax, yMin, yMax, zMin, zMax);

	drawPointCloud();

	if(pontosDe.size()>0){//so salva os arquivos se a nuvem de pontos estiver preenchida
		salvaArquivoEquivalenciaLand(pass); //salva arquivo com os landmarks
		salvaArquivoEquivalenciaDepth(pass); //salva arquivo com a nuvem de pontos
	}
  
	pass++; //contador de frame
	//pontosSh.clear();
	pontosDe.clear();
	imageDosLandmarks = Mat::zeros(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_8UC1); //zero a matriz de landmarks para começar outro frame
	imageAcertada = Mat::zeros(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_8UC4); //zero a matriz de landmarks para começar outro frame

    glFlush();
    glutSwapBuffers();
	fim = clock();
	printf("Frame%d: %lf segundos\n",pass, ((double)(fim - inicio)/CLOCKS_PER_SEC));
}

int init(){	
	
	NuiCreateSensorByIndex(0, &pSensor);
	//Inicializa o kinect
	pSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON );

	//Cria eventos de imagem e profundidade
	m_hNextImageFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
	m_pImageStreamHandle   = NULL;
	m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
	m_pDepthStreamHandle   = NULL;

	//Ativa câmera
	HRESULT hr;
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR , NUI_IMAGE_RESOLUTION_640x480 , 0 , 2 , m_hNextImageFrameEvent , &m_pImageStreamHandle );
	if( FAILED( hr ) ) 
		return -1;

	//Ativa sensor de profundidade
	hr = NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX , NUI_IMAGE_RESOLUTION_640x480 , 0 , 2 , m_hNextDepthFrameEvent , &m_pDepthStreamHandle );
	if( FAILED( hr ) ) 
		return -1;

	return 1;
}


void idle(){    
    glutPostRedisplay();
}


void reshape (int width, int height){
    FormWidth = width;
    FormHeight = height;
    glViewport (0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (glFovy, (GLfloat)width / (GLfloat)height,glZNear,glZFar);
    glMatrixMode (GL_MODELVIEW);
}

//Callback de evento de movimentação do mouse
void motion(int x, int y){
    int xDisp, yDisp;
    xDisp = x - xBegin;
    yDisp = y - yBegin;
    switch (mButton) {
    case GLUT_LEFT_BUTTON:
        azimuth += (float) xDisp/2.0;
        elevation -= (float) yDisp/2.0;
        break;
    case GLUT_MIDDLE_BUTTON:
        cameraX -= (float) xDisp/40.0;
        cameraY += (float) yDisp/40.0;
        break;
    case GLUT_RIGHT_BUTTON:
		cameraDistance += xDisp/40.0;
        break;
    }
    xBegin = x;
    yBegin = y;
}

//Callback de eventos do mouse
void mouse(int button, int state, int x, int y){ 
    if (state == GLUT_DOWN) {
        switch(button) {
        case GLUT_RIGHT_BUTTON:
        case GLUT_MIDDLE_BUTTON:
        case GLUT_LEFT_BUTTON:
            mButton = button;
            break;
        }
        xBegin = x;
        yBegin = y;
    }
}

//Callback de eventos de teclado
void Teclado(unsigned char key, int x, int y) {

    switch (key) {
        case 27:
            exit(0);
            break;
	}
}

int main(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(FormWidth, FormHeight);
    glutCreateWindow(argv[0]);
    glutReshapeFunc (reshape);
	glutKeyboardFunc(Teclado);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    init();
    glutMainLoop();
	
	NuiShutdown();
    return 0;
}
