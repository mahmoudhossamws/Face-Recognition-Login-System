#include "loginwindow.h"
#include "ui_loginwindow.h"
#include <QPushButton>
#include "mainwindow.h"
#include <QRegularExpression>
#include "success.h"
loginWindow::loginWindow(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::loginWindow)
{
    ui->setupUi(this);
    this->setFixedSize(this->size());

    this->setAutoFillBackground(true);
    QPixmap bg(":/main/build/Desktop_Qt_6_9_0_MinGW_64_bit-Debug/backgrounds/login.png");
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
            this, &loginWindow::onImageCaptured);

    // Start camera
    camera->start();
}

void loginWindow::onImageCaptured(int id, const QImage &preview)
{
    Q_UNUSED(id);

    QString baseDir = QCoreApplication::applicationDirPath(); // where .exe is
    QString pythonPath = "C:\\.venv\\Scripts\\python.exe";
    QString scriptPath = baseDir + "/findMatch.py";

    // Save or process the image
    QString filename = "loginAttempt.jpg";
    preview.save(filename);
    qDebug() << "Image saved to:" << filename;

    // Run the Python script
    QProcess *process = new QProcess(this);


    process->start(pythonPath, QStringList() << scriptPath);
    ui->status->setText("searching for matches");
    connect(process, &QProcess::readyReadStandardOutput, this, [=]() {
        QString output = process->readAllStandardOutput();
        qDebug() << "Output:" << output;

        if (output.contains("Best match:")) {
            QRegularExpression regex(R"(Best match:\s*(\w+)\s+sim:\s*([\d\.]+))");
            QRegularExpressionMatch match = regex.match(output);

            if (match.hasMatch()) {
                QString matchedName = match.captured(1);
                QString similarity = match.captured(2);
                qDebug() << "MATCH FOUND! Name:" << matchedName << "Similarity:" << similarity;

                Success* O = new Success(nullptr,matchedName);
                this->hide();
                O->show();
            }
        } else if (output.contains("No matching face found")) {
            qDebug() << "No match found.";
            ui->status->setText("No match");
        } else if (output.contains("Face not detected")) {
            qDebug() << "Face not detected.";
            ui->status->setText("Face not detected");
        }
    });

    connect(process, &QProcess::readyReadStandardError, this, [=]() {
        qDebug() << "Error:" << process->readAllStandardError();
    });
}
loginWindow::~loginWindow()
{
    delete ui;
}

void loginWindow::on_pushButton_clicked()
{
    MainWindow* O= new MainWindow();
    this->hide();
    O->show();
}

