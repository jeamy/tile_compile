#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

namespace tile_compile::gui {

class RunnerProcess : public QObject {
    Q_OBJECT

  public:
    explicit RunnerProcess(QObject *parent = nullptr);

    bool is_running() const;
    void start(const QStringList &cmd, const QString &cwd, const QString &stdin_data = QString());
    void stop();

  signals:
    void stdout_line(const QString &line);
    void stderr_line(const QString &line);
    void finished(int exit_code);

  private:
    void handle_stdout();
    void handle_stderr();

    QProcess *proc_ = nullptr;
    QString stdout_buffer_;
    QString stderr_buffer_;
};

}
