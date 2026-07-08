import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

// Firebase Config from User
const firebaseConfig = {
    apiKey: "AIzaSyD4iEMkXq3gJljeyF4wmF8UYAaqzDltSEA",
    authDomain: "gen-lang-client-0781062109.firebaseapp.com",
    projectId: "gen-lang-client-0781062109",
    storageBucket: "gen-lang-client-0781062109.firebasestorage.app",
    messagingSenderId: "638445321934",
    appId: "1:638445321934:web:43a5c267d4b67f37ab1a9f",
    measurementId: "G-SCYRJNF96C"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

let currentUser = null;
let allUsers = [];

const userTableBody = document.getElementById('userTableBody');
const searchInput = document.getElementById('searchInput');
const refreshBtn = document.getElementById('refreshBtn');
const adminEmailSpan = document.getElementById('adminEmail');
const logoutBtn = document.getElementById('logoutBtn');

// Guard
onAuthStateChanged(auth, async (user) => {
    if (user) {
        currentUser = user;
        adminEmailSpan.innerText = user.email;
        
        // Check if actually admin
        try {
            // Force refresh token to get latest custom claims
            const token = await user.getIdToken(true);
            const res = await fetch('/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                if (data.role !== 'admin') {
                    alert("Bạn không có quyền truy cập trang này!");
                    window.location.href = "/";
                } else {
                    loadUsers(); // Load data if valid admin
                }
            }
        } catch (e) {
            window.location.href = "/static/login.html";
        }
    } else {
        window.location.href = "/static/login.html";
    }
});

logoutBtn.addEventListener('click', () => {
    signOut(auth).then(() => {
        window.location.href = "/static/login.html";
    });
});

refreshBtn.addEventListener('click', () => loadUsers());

searchInput.addEventListener('input', () => {
    renderUsers(allUsers);
});

async function loadUsers() {
    userTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center;"><i class="fa-solid fa-spinner fa-spin"></i> Đang tải dữ liệu...</td></tr>`;
    try {
        const token = await currentUser.getIdToken(true);
        const res = await fetch('/admin/users', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
            allUsers = await res.json();
            renderUsers(allUsers);
        } else {
            throw new Error("Không thể lấy danh sách người dùng");
        }
    } catch (e) {
        console.error(e);
        userTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--danger-color);">Lỗi tải dữ liệu. Bạn có đủ quyền không?</td></tr>`;
    }
}

function renderUsers(users) {
    const query = searchInput.value.toLowerCase();
    const filtered = users.filter(u => u.email.toLowerCase().includes(query));

    if (filtered.length === 0) {
        userTableBody.innerHTML = `<tr><td colspan="4" style="text-align: center;">Không tìm thấy người dùng nào.</td></tr>`;
        return;
    }

    let html = '';
    filtered.forEach(u => {
        const roleBadge = u.admin 
            ? `<span class="badge admin"><i class="fa-solid fa-crown"></i> Admin</span>` 
            : `<span class="badge user"><i class="fa-solid fa-user"></i> User</span>`;
            
        const statusBadge = u.disabled 
            ? `<span class="badge banned"><i class="fa-solid fa-ban"></i> Bị khóa</span>` 
            : `<span class="badge active"><i class="fa-solid fa-check"></i> Hoạt động</span>`;

        html += `
            <tr>
                <td><strong>${u.email}</strong></td>
                <td>${roleBadge}</td>
                <td>${statusBadge}</td>
                <td>
                    <button class="action-btn" onclick="toggleRole('${u.uid}', ${!u.admin})" title="Đổi Quyền">
                        <i class="fa-solid ${u.admin ? 'fa-arrow-down' : 'fa-arrow-up'}"></i> ${u.admin ? 'Hạ quyền' : 'Lên Admin'}
                    </button>
                    <button class="action-btn ${u.disabled ? 'btn-primary' : 'btn-danger'}" onclick="toggleStatus('${u.uid}', ${!u.disabled})" title="Khóa/Mở khóa">
                        <i class="fa-solid ${u.disabled ? 'fa-unlock' : 'fa-lock'}"></i> ${u.disabled ? 'Mở khóa' : 'Khóa'}
                    </button>
                    <button class="action-btn" onclick="resetPassword('${u.uid}')" title="Reset Password">
                        <i class="fa-solid fa-key"></i> Đổi Pass
                    </button>
                </td>
            </tr>
        `;
    });
    userTableBody.innerHTML = html;
}

window.toggleRole = async (uid, isAdmin) => {
    if (!confirm(`Bạn muốn ${isAdmin ? 'Cấp quyền Admin' : 'Hạ quyền xuống User'} cho người dùng này?`)) return;
    try {
        const token = await currentUser.getIdToken();
        const res = await fetch(`/admin/users/${uid}/role`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ is_admin: isAdmin })
        });
        if (res.ok) loadUsers();
    } catch (e) {
        alert("Có lỗi xảy ra!");
    }
};

window.toggleStatus = async (uid, isDisabled) => {
    if (!confirm(`Bạn có chắc muốn ${isDisabled ? 'KHÓA' : 'MỞ KHÓA'} tài khoản này?`)) return;
    try {
        const token = await currentUser.getIdToken();
        const res = await fetch(`/admin/users/${uid}/toggle-status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ disabled: isDisabled })
        });
        if (res.ok) loadUsers();
    } catch (e) {
        alert("Có lỗi xảy ra!");
    }
};

window.resetPassword = async (uid) => {
    try {
        const token = await currentUser.getIdToken();
        const res = await fetch(`/admin/users/${uid}/reset-password`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (res.ok) {
            const data = await res.json();
            const linkInput = document.getElementById('resetLinkInput');
            linkInput.value = data.reset_link;
            document.getElementById('resetModal').style.display = 'flex';
        } else {
            alert("Lỗi khi tạo link reset password");
        }
    } catch (e) {
        alert("Có lỗi xảy ra!");
    }
};
