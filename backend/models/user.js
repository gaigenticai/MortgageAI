const bcrypt = require('bcrypt');
const pool = require('../config/database');

class User {
  static async create({ email, password, firstName, lastName, role = 'user' }) {
    const hashedPassword = await bcrypt.hash(password, 12);

    const query = `
      INSERT INTO users (email, password_hash, first_name, last_name, role)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id, email, first_name, last_name, role, created_at
    `;

    const values = [email, hashedPassword, firstName, lastName, role];

    try {
      const result = await pool.query(query, values);
      return result.rows[0];
    } catch (error) {
      if (error.code === '23505') { // unique violation
        throw new Error('User with this email already exists');
      }
      throw error;
    }
  }

  static async findByEmail(email) {
    const query = 'SELECT * FROM users WHERE email = $1 AND is_active = true';
    const result = await pool.query(query, [email]);
    return result.rows[0];
  }

  static async findById(id) {
    const query = 'SELECT id, email, first_name, last_name, role, created_at FROM users WHERE id = $1 AND is_active = true';
    const result = await pool.query(query, [id]);
    return result.rows[0];
  }

  static async validatePassword(plainPassword, hashedPassword) {
    return bcrypt.compare(plainPassword, hashedPassword);
  }

  static async updateLastLogin(id) {
    const query = 'UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = $1';
    await pool.query(query, [id]);
  }
}

module.exports = User;