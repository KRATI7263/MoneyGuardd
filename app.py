from flask import Flask, jsonify, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import io
import base64
import os
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

# Define model directory
model_dir = "notebook/"

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100),  nullable=False)
    email = db.Column(db.String(100),  nullable=False)
    password = db.Column(db.String(100))

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Transaction Model
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    description = db.Column(db.String(255))
    payment_method = db.Column(db.String(50), nullable=False)
    
    user = db.relationship('User', backref=db.backref('transactions', lazy=True))

with app.app_context():
    db.create_all()
# --------------------------------------------
# Routes for User Authentication
# --------------------------------------------

@app.route('/')
def index():
    return render_template('home.html')



@app.route('/signup', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)

    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')

# --------------------------------------------
# Stock Prediction Functions
# --------------------------------------------

def load_stock_model(stock_symbol):
    """Load the corresponding model for a given stock symbol."""
    model_path = os.path.join(model_dir, f'stock_price_model_{stock_symbol}.h5')
    return load_model(model_path) if os.path.exists(model_path) else None

def fetch_data(stock_symbol):
    """Fetch historical stock data for the given symbol."""
    try:
        stock_data = yf.download(stock_symbol, start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        return stock_data if not stock_data.empty else None
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def fetch_live_price(stock_symbol):
    """Fetch the current stock price for the given symbol."""
    try:
        live_data = yf.Ticker(stock_symbol)
        current_price = live_data.history(period='1d')
        return current_price['Close'].iloc[-1] if not current_price.empty else None
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return None

def preprocess_data(stock_data):
    """Preprocess the stock data for model prediction."""
    data = stock_data[['Open', 'Close']]
    return scaler.fit_transform(data)

def generate_graph(x_data, y_data_list, title, xlabel, ylabel, labels, colors):
    """Generate a graph and return it as a base64 encoded string."""
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    for y_data, label, color in zip(y_data_list, labels, colors):
        plt.plot(x_data, y_data, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# --------------------------------------------
# Routes for Stock Prediction
# --------------------------------------------

@app.route('/start')
def start():
    if 'email' not in session:
        return redirect('/login')  # Redirect to login if session is not active
    
    user = User.query.filter_by(email=session['email']).first()  # Fetch user data
    return render_template('stock_index.html', user=user)


@app.route('/finance')
def finance():
    if 'email' not in session:
        return redirect('/login')  # Redirect to login if session is not active
    
    user = User.query.filter_by(email=session['email']).first()  # Fetch user data
    return render_template('finanace_index.html', user=user)


@app.route('/finance_dashboard')
def finance_dashboard():
    if 'email' not in session:  
        return redirect('/login')  
    
    user = User.query.filter_by(email=session['email']).first()

    # Fetch only this user's transactions
    transactions = Transaction.query.filter_by(user_id=user.id).all()

    # Calculate totals
    total_upi = sum(t.amount for t in transactions if t.payment_method == "UPI")
    total_cash = sum(t.amount for t in transactions if t.payment_method == "Cash")
    total_amount = total_upi + total_cash

    # Debugging output
    print(f"Total UPI: {total_upi}, Total Cash: {total_cash}, Total Amount: {total_amount}")

    return render_template('finance_dash.html', user=user, total_upi=total_upi, total_cash=total_cash, total_amount=total_amount)


@app.route('/transactions', methods=['GET', 'POST'])
def transactions():
    if 'email' not in session:
        return redirect('/login')

    user = User.query.filter_by(email=session['email']).first()
    if not user:
        return redirect('/login')

    if request.method == 'POST':
        amount = float(request.form['amount'])
        category = request.form['category']
        date = request.form['date']
        description = request.form['description']
        payment_method = request.form['payment_method']

        new_transaction = Transaction(
            user_id=user.id, amount=amount, category=category,
            date=date, description=description, payment_method=payment_method
        )
        db.session.add(new_transaction)
        db.session.commit()
        return redirect('/transactions')

    user_transactions = Transaction.query.filter_by(user_id=user.id).all()
    
    return render_template('transaction.html', 
                           user=user, 
                           transactions=user_transactions)
                           
@app.route('/statistics')
def statistics():
    if 'email' not in session:
        return redirect('/login')
    
    user = User.query.filter_by(email=session['email']).first()
    user_transactions = Transaction.query.filter_by(user_id=user.id).all()
    
    total_spent = sum(t.amount for t in user_transactions)
    
    # Calculate spending per category
    expense_by_category = {}
    for transaction in user_transactions:
        expense_by_category[transaction.category] = expense_by_category.get(transaction.category, 0) + transaction.amount

    # Sort categories by highest spending
    top_spending_categories = sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)[:5]

    return render_template('statistic.html', user=user, total_spent=total_spent, 
                           expense_by_category=expense_by_category, 
                           top_spending_categories=top_spending_categories)


@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    if 'email' not in session:
        return redirect('/login')
    
    user = User.query.filter_by(email=session['email']).first()
    amount = float(request.form['amount'])
    category = request.form['category']
    date = request.form['date']
    description = request.form.get('description', '')
    payment_method = request.form['payment_method']
    
    new_transaction = Transaction(
        user_id=user.id, amount=amount, category=category,
        date=date, description=description, payment_method=payment_method
    )
    db.session.add(new_transaction)
    db.session.commit()
    
    return redirect('/transactions')

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    if 'email' not in session:
        return redirect('/login')
    
    transaction = Transaction.query.get(transaction_id)
    if transaction:
        db.session.delete(transaction)
        db.session.commit()
    
    return redirect('/transactions')

@app.route('/monthly_spending')
def monthly_spending():
    if 'email' not in session:
        return redirect('/login')
    
    user = User.query.filter_by(email=session['email']).first()
    current_month = datetime.now().strftime('%Y-%m')

    transactions = Transaction.query.filter(
        Transaction.user_id == user.id, 
        Transaction.date.like(f"{current_month}%")  # Ensure it matches YYYY-MM-DD format
    ).all()

    total_spent = sum(t.amount for t in transactions)

    return jsonify({"month": current_month, "total_spent": total_spent})

@app.route('/daily_spending')
def daily_spending():
    if 'email' not in session:
        return redirect('/login')
    
    user = User.query.filter_by(email=session['email']).first()
    today = datetime.now().strftime('%Y-%m-%d')

    transactions = Transaction.query.filter(
        Transaction.user_id == user.id, 
        Transaction.date == today  # Ensure it matches YYYY-MM-DD format
    ).all()

    total_spent = sum(t.amount for t in transactions)

    return jsonify({"date": today, "total_spent": total_spent})


@app.route('/start_predicting')
def start_predicting():
    if 'email' not in session:
        return redirect('/login')  # Redirect if user is not logged in

    user = User.query.filter_by(email=session['email']).first()
    return render_template('stock_index_1.html', user=user)

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return redirect('/login')

    user = User.query.filter_by(email=session['email']).first()
    stock_symbol = request.form['symbol']
    model = load_stock_model(stock_symbol)

    if model is None:
        return render_template('predict.html', error='Model not found for this stock.', user=user)

    stock_data = fetch_data(stock_symbol)
    if stock_data.empty:
        return render_template('predict.html', error='Invalid stock symbol or no data available', user=user)

    scaled_data = preprocess_data(stock_data)
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], last_60_days.shape[1]))

    predicted_scaled = model.predict(last_60_days)
    predicted_prices = scaler.inverse_transform(predicted_scaled)

    next_day_open_price = predicted_prices[0, 0]
    next_day_close_price = predicted_prices[0, 1]

    return render_template('predict.html', 
                           stock_symbol=stock_symbol.upper(),
                           next_day_open=next_day_open_price,
                           next_day_close=next_day_close_price,
                           user=user)

@app.route('/today_price', methods=['POST'])
def today_price():
    if 'email' not in session:
        return redirect('/login')

    user = User.query.filter_by(email=session['email']).first()
    stock_symbol = request.form['symbol']
    stock_data = fetch_data(stock_symbol)

    if stock_data.empty:
        return render_template('today.html', error='No data available for today.', user=user)

    today_open_price = stock_data['Open'].iloc[0]
    today_close_price = stock_data['Close'].iloc[-1]
    live_price = fetch_live_price(stock_symbol)

    close_chart_url = open_close_chart_url = None  # Initialize
    
    if not stock_data.empty:
        close_chart_url = generate_graph(
            stock_data.index, [stock_data['Close']],
            title=f"{stock_symbol} - Today's Close Price History",
            xlabel="Time", ylabel="Close Price",
            labels=["Close Price"], colors=['blue']
        )

        past_30_days = stock_data.tail(30)
        open_close_chart_url = generate_graph(
            past_30_days.index, [past_30_days['Open'], past_30_days['Close']],
            title=f"{stock_symbol} - Opening and Closing Prices (Last 30 Days)",
            xlabel="Date", ylabel="Price",
            labels=["Opening Price", "Closing Price"], colors=['green', 'red']
        )

    return render_template('today.html', 
                           stock_symbol=stock_symbol,
                           today_open=today_open_price,
                           today_close=today_close_price,
                           live_price=live_price,
                           close_chart_url=close_chart_url,
                           open_close_chart_url=open_close_chart_url,
                           user=user)

@app.route('/historical', methods=['POST'])
def historical():
    if 'email' not in session:
        return redirect('/login')

    user = User.query.filter_by(email=session['email']).first()
    stock_symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    if start_date >= end_date:
        return render_template('historical.html', error='Start date must be before end date.', user=user)

    historical_data = yf.download(stock_symbol, start=start_date, end=end_date)

    historical_chart_url = None  # Initialize variable

    if not historical_data.empty:
        historical_chart_url = generate_graph(
            historical_data.index, 
            [historical_data['Close'], historical_data['Open']],
            title=f"{stock_symbol} - Historical Open and Close Prices ({start_date} to {end_date})",
            xlabel="Date", 
            ylabel="Price",
            labels=["Closing Price", "Opening Price"],
            colors=['red', 'green']
        )

    return render_template('historical.html', 
                           stock_symbol=stock_symbol.upper(),
                           start_date=start_date,
                           end_date=end_date,
                           historical_chart_url=historical_chart_url,
                           user=user)

if __name__ == '__main__':
    app.run(debug=True)
