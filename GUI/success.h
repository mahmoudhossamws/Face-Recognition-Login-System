#ifndef SUCCESS_H
#define SUCCESS_H

#include <QWidget>

namespace Ui {
class Success;
}

class Success : public QWidget
{
    Q_OBJECT

public:
   explicit Success(QWidget *parent = nullptr, QString name="who");
    ~Success();

   private slots:
   void on_pushButton_clicked();

   private:
    Ui::Success *ui;
};

#endif // SUCCESS_H
