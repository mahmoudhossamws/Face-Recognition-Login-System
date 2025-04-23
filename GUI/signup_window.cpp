#include "signup_window.h"
#include "ui_signup_window.h"
#include <QPushButton>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include "mainwindow.h"
signUp_window::signUp_window(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::signUp_window)
{
    ui->setupUi(this);

    this->setAutoFillBackground(true);
    QPixmap bg(":/main/build/Desktop_Qt_6_9_0_MinGW_64_bit-Debug/backgrounds/signup.png");
    bg = bg.scaled(this->size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(bg));
    this->setPalette(palette);

    // Initialize camera components
    camera = new QCamera(this);
    viewfinder = new QVideoWidget(this);
    session = new QMediaCaptureSession(this);
    imageCapture = new QImageCapture(this);

    // Set up camera session
    session->setCamera(camera);
    session->setVideoOutput(viewfinder);
    session->setImageCapture(imageCapture);

    // Add the camera viewfinder to the layout from the UI
    ui->verticalLayout->insertWidget(0, viewfinder); // Assuming you have a verticalLayout in your UI

    // Optional: set size for camera view
    viewfinder->setFixedSize(320, 240);

    // Connect UI button signals
    connect(ui->snapButton, &QPushButton::clicked, this, [this]() {
        imageCapture->capture();
    });

    connect(imageCapture, &QImageCapture::imageCaptured,
            this, &signUp_window::onImageCaptured);

    // Start camera
    camera->start();
}


signUp_window::~signUp_window()
{
    delete ui;
}

void signUp_window::onImageCaptured(int id, const QImage &preview)
{
    Q_UNUSED(id);

    QString baseDir = QCoreApplication::applicationDirPath(); // where .exe is
    QString pythonPath = "C:\\.venv\\Scripts\\python.exe";
    QString scriptPath = baseDir + "/newEmbedding.py";

    // Save or process the image
    QString filename = "person.jpg";
    preview.save(filename);
    qDebug() << "Image saved to:" << filename;

   QString userName=ui->userNameEdit->text();
  // Run the Python script
    QProcess *process = new QProcess(this);

   ui->status->setText("adding "+ userName+ " face \n to the database");

    process->start(pythonPath, QStringList() << scriptPath<< userName);

    // Optional: connect to signals for finished or error output
    connect(process, &QProcess::readyReadStandardOutput, [=]() {
         QString output = process->readAllStandardOutput();
        qDebug() << "Output:" << output;
        if (output.contains("success", Qt::CaseInsensitive)) {
            ui->status->setText(userName + "'s face added successfully to the database.");
        } else if (output.contains("Face not detected", Qt::CaseInsensitive)) {
            ui->status->setText("Face not detected. Please try again.");
        }
    });
    connect(process, &QProcess::readyReadStandardError, [=]() {qDebug() << "Error:" << process->readAllStandardError();
    });

}

void signUp_window::on_pushButton_clicked()
{
    MainWindow* O= new MainWindow();
    this->hide();
    O->show();
}

