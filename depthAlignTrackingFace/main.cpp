#include <stdlib.h>
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

#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEIGHT 480
#define KINECT_DEPTH_WIDTH 640
#define KINECT_DEPTH_HEIGHT 480

HANDLE m_hNextImageFrameEvent;
HANDLE m_hNextDepthFrameEvent;
HANDLE m_pImageStreamHandle;
HANDLE m_pDepthStreamHandle;
INuiSensor* pSensor;
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
Mat DepthinRGB(KINECT_IMAGE_HEIGHT,KINECT_IMAGE_WIDTH,CV_16UC1);
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
	float b;};


pontosShape pShape;
pontosDepth pDepth;
vector<pontosShape> pontosSh;
vector<pontosDepth> pontosDe;

cv::Mat shapeG,conG,triG,visiG;

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
		arq.open(fileName+".txt");
				
		for(size_t i = 0; i<pontosSh.size();i++){
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
		arq.open(fileName+".txt");

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

	c = CV_RGB(0,255,0);
    //if (i<=26) c = CV_RGB(255,0,0);			   // Contorno facial 26
    //else if (i>26&&i<=30) c = CV_RGB(0,128,0); // Nariz 4
    //else if (i>30&&i<=35) c = CV_RGB(0,255,0); // Parte abaixo nariz 5
    //else if (i>35&&i<=47) c = CV_RGB(0,0,255); // olhos 12
    //else c = CV_RGB(0,0,128);                  // Boca 18

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

void createLandMarkImage(){

	int i, n = shapeG.rows/2;
	cv::Point p1;
 
	for(i = 0; i < n; i++){
		
		p1 = cv::Point(shapeG.at<double>(i,0),shapeG.at<double>(i+n,0));
		imageDosLandmarks.at<uchar>(p1.y,p1.x) = 255;
	}
	//imshow("out", imageDosLandmarks);

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

void drawPointCloud(Mat &rgbImage,Mat &pointCloud_XYZ, Mat &shapis){
    static int x,y;

    glPointSize(1);
    glBegin(GL_POINTS);

		uchar *p = (unsigned char*)(rgbImage.data); //ponteiro para os dados da imagem RGB
		uchar *land = (unsigned char*)(shapis.data); //ponteiro para os dados da imagem RGB
		Point3f *point = (Point3f*)pointCloud_XYZ.data; //ponteiro para os dados de profundidade XYZ
		
		LONG colorX,colorY; //ponto na matriz onde esta determinada cor
		float rN,gN,bN;
		float r,g,b;
		float rLand, gLand, bLand;
		int cont = 0;

		for(y = 0;y < pointCloud_XYZ.rows;y++){
			for(x = 0;x < pointCloud_XYZ.cols;x++,point++){				
				
				NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480,NUI_IMAGE_RESOLUTION_640x480,NULL,x,y,0,&colorX,&colorY); //tem como saida a linha e coluna de cor do pixel
				
				if(0 <= colorX && colorX < shapis.cols && 0 <= colorY && colorY < shapis.rows){	
						
					if(shapis.at<uchar>(colorY,colorX) == 255){
						pShape.pX = point->x;
						pShape.pY = point->y;
						pShape.pZ = point->z;
						pontosSh.push_back(pShape);
						cont++;
					}
				}				

				if(0 <= colorX && colorX <= rgbImage.cols && 0 <= colorY && colorY <= rgbImage.rows){						

					r = (p[colorY * rgbImage.step + colorX * rgbImage.channels()])  ;//pega o pixel vermelho da imagem rgb
					g = (p[colorY * rgbImage.step + colorX * rgbImage.channels()+1]);//pega o pixel verde da imagem rgb
					b = (p[colorY * rgbImage.step + colorX * rgbImage.channels()+2]);//pega o pixel azul da imagem rgb
				
					rN = r/255;//converte o valor do pixel vermelho para valores entre 0 e 1
					gN = g/255;//converte o valor do pixel verde para valores entre 0 e 1
					bN = b/255;//converte o valor do pixel azul para valores entre 0 e 1

					glColor3f(rN,gN,bN);
					glVertex3f(point->x,point->y,point->z); //desenha os pontos na tela
				}				

				if(point->z<1.1){//salva os pontos e respectivas cores no vetor de pontos	
					pDepth.pX = point->x;
					pDepth.pY = point->y;
					pDepth.pZ = point->z;
					pDepth.r = r;
					pDepth.g = g;
					pDepth.b = b;
					pontosDe.push_back(pDepth);
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

	if(GetImage(image,m_hNextImageFrameEvent,m_pImageStreamHandle)==-1) //retorna imagem RGB
		return;

	if(GetImage(depthPure,m_hNextDepthFrameEvent,m_pDepthStreamHandle)==-1) //retorna image de profundidade
		return;

	imageComLand = image.clone();

	detectaFace(imageComLand); //detecto os landmarks da face(o que me interessa é shapeG(os pontos))

	createLandMarkImage(); //converto shapeG para imagem
	
	//transforma os pontos de depth em uma nuvem de pontos 3D
    retrievePointCloudMap(depthPure,pointCloud_XYZ);		

	polarview();

    //imshow("depth",depthPure);
	
	//converte de BGR para RGB
    cvtColor(image,image,CV_RGBA2BGRA);  

    //desenha a nuvem de pontos
	drawPointCloud(image,pointCloud_XYZ, imageDosLandmarks);

	
	salvaArquivoEquivalenciaLand(pass); //salva arquivo com os landmarks
	//salvaArquivoEquivalenciaDepth(pass); //salva arquivo com a nuvem de pontos
  
	pass++;
	pontosSh.clear();
	pontosDe.clear();
	imageDosLandmarks = Mat::zeros(KINECT_DEPTH_HEIGHT,KINECT_DEPTH_WIDTH,CV_8UC1);
 
    glFlush();
    glutSwapBuffers();
	fim = clock();
	printf("Frame%d: %lf\n",pass, ((double)(fim - inicio)/CLOCKS_PER_SEC));
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

//evento de movimentação do mouse
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

//evento de clique do mouse
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

int main(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(FormWidth, FormHeight);
    glutCreateWindow(argv[0]);
    glutReshapeFunc (reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    init();
    glutMainLoop();
	
	NuiShutdown();
    return 0;
}