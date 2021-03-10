import os
import datetime as dt
import pymysql
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash

app = Flask(__name__)

app.config.update(dict(
        host='localhost',
        user='root',
        password='123456',
        port=3306,
))
app.config.from_envvar('TC_SETTINGS', silent=True)


def connect_db():
    # Connect to the TC database
    db = pymysql.connect(host=app.config['host'], user=app.config['user'],
                         password=app.config['password'], port=app.config['port'])
    return db


def get_db():
    # Opens a new connection to the TC database
    if not hasattr(g, 'mysql_db'):
        g.mysql_db = connect_db()
    return g.mysql_db


def init_db(database, table1, table2):
    # Create the TC database tables
    cursor = get_db().cursor()
    with app.app_context():
        exist = 'show databases;'
        cursor.execute(exist)
        all_base = list(cursor.fetchall())
        if ('{}'.format(database),) in all_base:
            restore = 'use {};'.format(database)
            cursor.execute(restore)
        else:
            database = 'CREATE DATABASE {} DEFAULT CHARACTER SET utf8'.format(database)
            cursor.execute(database)
            cursor.execute('use {};'.format(database))
        cursor.execute('DROP {};DROP {};'.format(table1, table2))
        cursor.execute('CREATE TABLE IF NOT EXISTS users (username VARCHAR(255) NOT NULL,'
                       'password VARCHAR(255) NOT NULL, userid INT NOT NULL, PRIMARY KEY (userid))')
        cursor.execute('CREATE TABLE IF NOT EXISTS comments (userid INT NOT NULL, username VARCHAR(255) NOT NULL'
                       'time DATETIME NOT NULL, comments VARCHAR(1023) NOT NULL')


@app.teardown_appcontext
def close_db(error):
    # Close the TC database at the end of the request
    if hasattr(g, 'mysql_db'):
        g.mysql_db.close()


@app.route('/')
def show_entries(table):
    cursor = get_db().cursor()
    query = 'SELECT comment, user, time from comments ORDER BY userid DESC'
    cursor.execute(query)
    comments = cursor.fetchall()
    return render_template('show_entries.html', comments=comments)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        db = get_db()
        cursor = db.cursor()
        username = request.form['username']
        password = request.form['password']
        user_id = hash(username)
        if username == "" or password == "":
            error = 'Provide both a username and a password'
        else:
            data = 'INSERT INTO users (name, password, userid) values (%s,%s,%s)'
            try:
                cursor.execute(data, (username, password, user_id))
                db.commit()
                session['logged_in'] = True
                flash('You were sussessfully registered.')
                app.config.update(dict(USERNAME=request.form['password']))
                return redirect(url_for('show_entries'))
            except pymysql.Error as e:
                db.rollback()
                error = 'username exists'
                return render_template('register.html', error=error)
    return render_template('register.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Logs in a user
    error = None
    if request.method == 'POST':
        db = get_db()
        cursor = db.cursor()
        try:
            query = 'SELECT userid FROM users WHERE name=(%s) and password=(%s)'
            id = cursor.execute(query,(request.form['username'], request.form['password'])).fetchone()
            session['logged_in'] = True
            flash('You are now logged in.')
            app.config.update(dict(USERNAME=request.form['username']))
            return redirect(url_for('show_entries'))
        except:
            error = 'User not found or wrong password.'
    return render_template('login.htnl', error=error)


@app.route('/add', method=['POST'])
def add_entry():
    if not session.get('logged_in'):
        abort(401)
    db = get_db()
    cursor = db.cursor()
    now = dt.datetime.now()
    comment = 'insert into commects (comment, user, time) values (%s,%s,%s)'
    try:
        cursor.execute(comment, (request.form['text'], app.config['USERNAME'], str(now)))
        db.commit()
        flash('Your comment was successfully added.')
        return redirect(url_for('show_entries'))
    except pymysql.Error as e:
        db.rollback()


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_entries'))

if __name__ == 'main':
    init_db()
    app.run()

































