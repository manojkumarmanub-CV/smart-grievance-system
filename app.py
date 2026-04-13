import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, session, url_for, Response
from werkzeug.utils import secure_filename
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_secret")

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///complaints.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Upload configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# Email configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Department login credentials
USERS = {
    os.getenv("ADMIN_USERNAME", "admin"): {
        "password": os.getenv("ADMIN_PASSWORD", "admin123"),
        "department": "All"
    },
    os.getenv("WATER_USERNAME", "water"): {
        "password": os.getenv("WATER_PASSWORD", "water123"),
        "department": "Water Supply Department"
    },
    os.getenv("ELECTRICITY_USERNAME", "electricity"): {
        "password": os.getenv("ELECTRICITY_PASSWORD", "elec123"),
        "department": "Electricity Board"
    },
    os.getenv("PWD_USERNAME", "pwd"): {
        "password": os.getenv("PWD_PASSWORD", "pwd123"),
        "department": "Public Works Department"
    },
    os.getenv("MUNICIPAL_USERNAME", "municipal"): {
        "password": os.getenv("MUNICIPAL_PASSWORD", "muni123"),
        "department": "Municipal Corporation"
    },
    os.getenv("GENERAL_USERNAME", "general"): {
        "password": os.getenv("GENERAL_PASSWORD", "general123"),
        "department": "General Administration"
    }
}


class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    text = db.Column(db.String(500), nullable=False)
    sentiment = db.Column(db.String(50))
    urgency = db.Column(db.String(50))
    priority = db.Column(db.String(50), default="Low")
    department = db.Column(db.String(100))
    status = db.Column(db.String(50), default="Pending")
    remark = db.Column(db.String(500), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(200))


urgent_keywords = ["immediately", "urgent", "asap", "emergency", "danger", "serious"]


def send_email_notification(to_email, subject, body):
    if not to_email or not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Email skipped: missing email configuration.")
        return False

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())

        return True
    except Exception as e:
        print("Email sending failed:", e)
        return False


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_urgency(text):
    text = text.lower()
    for word in urgent_keywords:
        if word in text:
            return "High"
    return "Normal"


def calculate_priority(text):
    text = text.lower()
    score = 0

    critical_words = ["emergency", "fire", "live wire", "death", "accident", "critical"]
    high_words = ["danger", "urgent", "immediately", "serious", "broken", "collapse"]
    medium_words = ["delay", "issue", "problem", "not working", "complaint", "repair"]

    for word in critical_words:
        if word in text:
            score += 4

    for word in high_words:
        if word in text:
            score += 3

    for word in medium_words:
        if word in text:
            score += 2

    if score >= 8:
        return "Critical"
    elif score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def classify_department(text):
    text = text.lower()

    if "road" in text or "pothole" in text or "bridge" in text:
        return "Public Works Department"
    elif "water" in text or "drainage" in text or "pipe" in text or "sewage" in text:
        return "Water Supply Department"
    elif "electricity" in text or "power" in text or "current" in text or "wire" in text:
        return "Electricity Board"
    elif "garbage" in text or "waste" in text or "clean" in text or "dog" in text or "dogs" in text:
        return "Municipal Corporation"
    else:
        return "General Administration"


def generate_ticket_id():
    last = Complaint.query.order_by(Complaint.id.desc()).first()
    if not last:
        number = 1
    else:
        number = last.id + 1
    return f"GRV-2026-{number:03d}"


def find_similar_complaint(new_text):
    complaints = Complaint.query.all()

    if not complaints:
        return None, 0

    new_text = (new_text or "").strip().lower()

    if len(new_text) < 20:
        return None, 0

    existing_texts = []
    valid_complaints = []

    for complaint in complaints:
        existing_text = (complaint.text or "").strip().lower()
        if existing_text:
            existing_texts.append(existing_text)
            valid_complaints.append(complaint)

    if not existing_texts:
        return None, 0

    all_texts = existing_texts + [new_text]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        max_score = similarity_scores.max()

        if max_score >= 0.80:
            index = similarity_scores.argmax()
            return valid_complaints[index], float(max_score)

        return None, float(max_score)

    except Exception as e:
        print("Similarity check failed:", e)
        return None, 0


def calculate_sla_status(created_at):
    now = datetime.utcnow()
    diff = now - created_at
    hours = diff.total_seconds() / 3600

    if hours >= 48:
        return "Escalated"
    elif hours >= 24:
        return "Warning"
    else:
        return "Normal"


@app.route("/")
def entry_portal():
    return render_template("index.html")


@app.route("/public", methods=["GET", "POST"])
def public_home():
    sentiment = ""
    urgency = ""
    priority = ""
    department = ""
    ticket_id = ""

    duplicate_warning = ""
    duplicate_ticket = ""
    show_duplicate_options = False
    file_error = ""
    success_message = ""

    submitted = request.args.get("submitted")
    success_ticket_id = request.args.get("ticket_id")

    if submitted == "1" and success_ticket_id:
        complaint = Complaint.query.filter_by(ticket_id=success_ticket_id).first()
        if complaint:
            sentiment = complaint.sentiment
            urgency = complaint.urgency
            priority = complaint.priority
            department = complaint.department
            ticket_id = complaint.ticket_id
            success_message = "Complaint submitted successfully."

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        complaint_text = request.form.get("complaint", "").strip()
        force_submit = request.form.get("force_submit", "").strip()

        uploaded_file = request.files.get("evidence")
        saved_filename = None

        if not name or not phone or not email or not complaint_text:
            return render_template(
                "public.html",
                file_error="All fields are required.",
                name=name,
                phone=phone,
                email=email,
                complaint_text=complaint_text
            )

        if force_submit != "yes":
            similar_complaint, similarity_score = find_similar_complaint(complaint_text)

            if similar_complaint:
                duplicate_warning = (
                    f"Similar complaint detected (Ticket ID: {similar_complaint.ticket_id}) "
                    f"Similarity Score: {similarity_score:.2f}"
                )
                duplicate_ticket = similar_complaint.ticket_id
                show_duplicate_options = True

                return render_template(
                    "public.html",
                    duplicate_warning=duplicate_warning,
                    duplicate_ticket=duplicate_ticket,
                    show_duplicate_options=show_duplicate_options,
                    name=name,
                    phone=phone,
                    email=email,
                    complaint_text=complaint_text
                )

        if uploaded_file and uploaded_file.filename != "":
            if allowed_file(uploaded_file.filename):
                filename = secure_filename(uploaded_file.filename)
                saved_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                uploaded_file.save(os.path.join(app.config["UPLOAD_FOLDER"], saved_filename))
            else:
                file_error = "Only PNG, JPG, JPEG, and PDF files are allowed."
                return render_template(
                    "public.html",
                    file_error=file_error,
                    name=name,
                    phone=phone,
                    email=email,
                    complaint_text=complaint_text
                )

        blob = TextBlob(complaint_text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        urgency = detect_urgency(complaint_text)
        priority = calculate_priority(complaint_text)
        department = classify_department(complaint_text)
        ticket_id = generate_ticket_id()

        new_complaint = Complaint(
            ticket_id=ticket_id,
            name=name,
            phone=phone,
            email=email,
            text=complaint_text,
            sentiment=sentiment,
            urgency=urgency,
            priority=priority,
            department=department,
            status="Pending",
            remark=f"Complaint submitted to {department}.",
            filename=saved_filename
        )

        db.session.add(new_complaint)
        db.session.commit()

        email_subject = "Complaint Registered Successfully"
        email_body = f"""
Dear {name},

Your complaint has been registered successfully.

Tracking ID: {ticket_id}
Department: {department}
Urgency: {urgency}
Priority: {priority}
Status: Pending

Thank you,
AI Grievance Management System
"""
        send_email_notification(email, email_subject, email_body)

        return redirect(url_for("public_home", submitted="1", ticket_id=ticket_id))

    return render_template(
        "public.html",
        sentiment=sentiment,
        urgency=urgency,
        priority=priority,
        department=department,
        ticket_id=ticket_id,
        duplicate_warning=duplicate_warning,
        duplicate_ticket=duplicate_ticket,
        show_duplicate_options=show_duplicate_options,
        file_error=file_error,
        success_message=success_message
    )


@app.route("/track", methods=["GET", "POST"])
def track_complaint():
    complaint = None
    complaints = None
    error = ""

    prefill_ticket_id = request.args.get("ticket_id", "")

    if request.method == "POST":
        ticket_id = request.form.get("ticket_id", "").strip()
        phone = request.form.get("phone", "").strip()

        if not ticket_id and not phone:
            error = "Please enter Tracking ID or Phone Number."
        elif ticket_id and phone:
            complaint = Complaint.query.filter_by(ticket_id=ticket_id, phone=phone).first()
            if not complaint:
                error = "No complaint found with the given Tracking ID and Phone Number."
        elif ticket_id:
            complaint = Complaint.query.filter_by(ticket_id=ticket_id).first()
            if not complaint:
                error = "No complaint found with the given Tracking ID."
        elif phone:
            complaints = Complaint.query.filter_by(phone=phone).order_by(Complaint.id.desc()).all()
            if not complaints:
                error = "No complaints found with the given Phone Number."

    return render_template(
        "track.html",
        complaint=complaint,
        complaints=complaints,
        error=error,
        prefill_ticket_id=prefill_ticket_id
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = USERS.get(username)

        if user and user["password"] == password:
            session["logged_in"] = True
            session["username"] = username
            session["department"] = user["department"]
            return redirect(url_for("admin_dashboard"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/admin")
def admin_dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    user_department = session.get("department", "All")

    status_filter = request.args.get("status", "")
    department_filter = request.args.get("department", "")
    priority_filter = request.args.get("priority", "")
    search = request.args.get("search", "").strip()

    query = Complaint.query

    if user_department != "All":
        query = query.filter_by(department=user_department)

    if status_filter:
        query = query.filter_by(status=status_filter)

    if user_department == "All" and department_filter:
        query = query.filter_by(department=department_filter)

    if priority_filter:
        query = query.filter_by(priority=priority_filter)

    if search:
        query = query.filter(
            Complaint.text.ilike(f"%{search}%") |
            Complaint.name.ilike(f"%{search}%") |
            Complaint.ticket_id.ilike(f"%{search}%")
        )

    complaints = query.all()

    priority_order = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    complaints.sort(
        key=lambda x: (
            priority_order.get(x.priority, 0),
            x.created_at
        ),
        reverse=True
    )

    stats_query = Complaint.query
    if user_department != "All":
        stats_query = stats_query.filter_by(department=user_department)

    total = stats_query.count()
    pending = stats_query.filter_by(status="Pending").count()
    in_progress = stats_query.filter_by(status="In Progress").count()
    resolved = stats_query.filter_by(status="Resolved").count()
    rejected = stats_query.filter_by(status="Rejected").count()

    critical_count = stats_query.filter_by(priority="Critical").count()
    high_count = stats_query.filter_by(priority="High").count()
    medium_count = stats_query.filter_by(priority="Medium").count()
    low_count = stats_query.filter_by(priority="Low").count()

    department_counts_query = Complaint.query
    if user_department != "All":
        department_counts_query = department_counts_query.filter_by(department=user_department)

    public_works_count = department_counts_query.filter_by(department="Public Works Department").count()
    water_supply_count = department_counts_query.filter_by(department="Water Supply Department").count()
    electricity_count = department_counts_query.filter_by(department="Electricity Board").count()
    municipal_count = department_counts_query.filter_by(department="Municipal Corporation").count()
    general_count = department_counts_query.filter_by(department="General Administration").count()

    return render_template(
        "admin.html",
        complaints=complaints,
        total=total,
        pending=pending,
        in_progress=in_progress,
        resolved=resolved,
        rejected=rejected,
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        status_filter=status_filter,
        department_filter=department_filter,
        priority_filter=priority_filter,
        search=search,
        public_works_count=public_works_count,
        water_supply_count=water_supply_count,
        electricity_count=electricity_count,
        municipal_count=municipal_count,
        general_count=general_count,
        calculate_sla_status=calculate_sla_status,
        user_department=user_department
    )


@app.route("/update_status/<int:id>", methods=["POST"])
def update_status(id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    complaint = Complaint.query.get_or_404(id)
    user_department = session.get("department", "All")

    if user_department == "All" or complaint.department == user_department:
        complaint.status = request.form.get("status", complaint.status)
        complaint.remark = request.form.get("remark", "").strip()
        db.session.commit()

        email_subject = f"Complaint Status Updated - {complaint.ticket_id}"
        email_body = f"""
Dear {complaint.name},

Your complaint status has been updated.

Tracking ID: {complaint.ticket_id}
New Status: {complaint.status}
Remark: {complaint.remark}
Department: {complaint.department}
Priority: {complaint.priority}

Thank you,
AI Grievance Management System
"""
        send_email_notification(complaint.email, email_subject, email_body)

    return redirect(url_for("admin_dashboard"))


@app.route("/export")
def export_csv():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    user_department = session.get("department", "All")
    query = Complaint.query

    if user_department != "All":
        query = query.filter_by(department=user_department)

    complaints = query.order_by(Complaint.id.desc()).all()

    def generate():
        yield "Ticket ID,Name,Phone,Email,Complaint,Sentiment,Urgency,Priority,Department,Status,Remark,Created At,Filename\n"
        for c in complaints:
            row = (
                f'"{c.ticket_id}",'
                f'"{c.name}",'
                f'"{c.phone}",'
                f'"{c.email}",'
                f'"{c.text}",'
                f'"{c.sentiment}",'
                f'"{c.urgency}",'
                f'"{c.priority}",'
                f'"{c.department}",'
                f'"{c.status}",'
                f'"{c.remark}",'
                f'"{c.created_at.strftime("%d %b %Y %H:%M")}",'
                f'"{c.filename if c.filename else ""}"\n'
            )
            yield row

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=complaints.csv"}
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)