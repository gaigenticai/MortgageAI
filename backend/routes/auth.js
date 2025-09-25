const User = require('../models/user');
const { authSchemas, validate } = require('../utils/validation');

async function authRoutes(fastify, options) {
  // Login endpoint
  fastify.post('/api/auth/login', {
    schema: {
      body: authSchemas.login
    },
    preHandler: fastify.rateLimit({
      max: 5,
      timeWindow: '5 minutes',
      keyGenerator: (req) => req.body.email || req.ip
    })
  }, async (request, reply) => {
    try {
      const { email, password } = request.body;

      // Find user
      const user = await User.findByEmail(email);
      if (!user) {
        fastify.log.warn(`Failed login attempt for non-existent user: ${email}`);
        return reply.code(401).send({
          error: 'Invalid credentials',
          message: 'Email or password is incorrect'
        });
      }

      // Validate password
      const isValidPassword = await User.validatePassword(password, user.password_hash);
      if (!isValidPassword) {
        fastify.log.warn(`Failed login attempt for user: ${email}`);
        return reply.code(401).send({
          error: 'Invalid credentials',
          message: 'Email or password is incorrect'
        });
      }

      // Update last login
      await User.updateLastLogin(user.id);

      // Generate JWT
      const token = fastify.jwt.sign({
        id: user.id,
        email: user.email,
        role: user.role
      }, {
        expiresIn: process.env.JWT_EXPIRES_IN || '24h'
      });

      fastify.log.info(`Successful login for user: ${email}`);

      return {
        token,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role
        }
      };
    } catch (error) {
      fastify.log.error('Login error:', error);
      return reply.code(500).send({
        error: 'Internal server error',
        message: 'An error occurred during login'
      });
    }
  });

  // Register endpoint
  fastify.post('/api/auth/register', {
    schema: {
      body: authSchemas.register
    },
    preHandler: fastify.rateLimit({
      max: 3,
      timeWindow: '15 minutes',
      keyGenerator: (req) => req.body.email || req.ip
    })
  }, async (request, reply) => {
    try {
      const { email, password, firstName, lastName } = request.body;

      // Create user
      const user = await User.create({
        email,
        password,
        firstName,
        lastName
      });

      fastify.log.info(`New user registered: ${email}`);

      return {
        message: 'User registered successfully',
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role
        }
      };
    } catch (error) {
      if (error.message === 'User with this email already exists') {
        return reply.code(409).send({
          error: 'Conflict',
          message: error.message
        });
      }

      fastify.log.error('Registration error:', error);
      return reply.code(500).send({
        error: 'Internal server error',
        message: 'An error occurred during registration'
      });
    }
  });

  // Profile endpoint
  fastify.get('/api/auth/profile', {
    preHandler: fastify.auth(['authenticate'])
  }, async (request, reply) => {
    try {
      const user = await User.findById(request.user.id);
      if (!user) {
        return reply.code(404).send({
          error: 'Not found',
          message: 'User not found'
        });
      }

      return {
        id: user.id,
        email: user.email,
        firstName: user.first_name,
        lastName: user.last_name,
        role: user.role,
        createdAt: user.created_at
      };
    } catch (error) {
      fastify.log.error('Profile fetch error:', error);
      return reply.code(500).send({
        error: 'Internal server error',
        message: 'An error occurred while fetching profile'
      });
    }
  });
}

module.exports = authRoutes;