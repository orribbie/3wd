

  
//  Jetson <──USB Serial──> Arduino (direct, no ESP/XIAO mux)
//
//  Commands (from Jetson via USB serial):
//    g x y theta   — goto position (grid units, degrees)
//    stop          — halt all motors
//    z             — zero pose + yaw
//    s             — status report
//    y             — quick yaw query
//    log 1 label   — start CSV telemetry stream
//    log 0         — stop  CSV telemetry stream
//
//  Telemetry lines (when logging) are prefixed "TEL," for
//  easy parsing on the Jetson side.
// ============================================================

#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_INA260.h>
#include <CytronMotorDriver.h>
#include <Encoder.h>
#include <Wire.h>
#include <math.h>

// ─── ROBOT CONFIG ────────────────────────────────────────────
const float GRID_UNIT_MM      = 100.0f;
const float WHEEL_RADIUS_MM   = 60.0f;
const int   ENCODER_PPR_MOTOR = 4;
const float GEAR_RATIO        = 41.6f;
const float COUNTS_PER_REV    = ENCODER_PPR_MOTOR * GEAR_RATIO * 16.0f;
const float KP_POS            = 0.2f;
int         PWM_POS_MAX       = 100;   // mutable — updated by  spd  command
const long  COUNT_TOL         = 20;
const float YAW_KP            = 3.0f;
const float YAW_DEADBAND      = 3.0f;
int         PWM_YAW_MAX       = 50;    // mutable — updated by  spd  command
const int   PWM_MOVE_MIN      = 35;    // minimum PWM to overcome static friction

const uint8_t PWM_M1 = 6,  DIR_M1 = 7;
const uint8_t PWM_M2 = 4,  DIR_M2 = 5;
const uint8_t PWM_M3 = 8,  DIR_M3 = 10;
const int ENC1_A = 13, ENC1_B = 3;
const int ENC2_A = 2,  ENC2_B = 9;
const int ENC3_A = 18, ENC3_B = 11;

const uint8_t ADXL345_I2C_ADDR = 0x53;

const unsigned long TELEMETRY_DT_MS = 200;
const float ROBOT_RADIUS_MM         = 150.0f;
const float DT_MIN_S                = 0.002f;
const float DT_MAX_S                = 0.05f;
const float ACC_DEADBAND_MPS2       = 0.12f;
const float ACC_LPF_FC_HZ          = 6.0f;
const float VEL_LEAK_PER_S         = 0.69f;
const float RLS_ETA                = 0.99f;
const float RLS_VMIN_MPS           = 0.03f;
const float ALPHA_MIN              = 0.93f;
const float ALPHA_MAX              = 1.00f;
const float P0                     = 50.0f;
const float ALPHA_LPF_FC_HZ        = 1.5f;
const float I_BASELINE_LPF_FC_HZ   = 0.5f;

// ─── TELEMETRY / LOG STATE ───────────────────────────────────
static bool          LOG_ON         = false;
static char          LOG_LABEL[16]  = "none";
const unsigned long  LOG_DT_MS      = 40;
static unsigned long lastLogMs      = 0;
static bool          autoGotoLog    = false;
static unsigned long autoLogStartMs = 0;
const unsigned long  AUTO_LOG_DUR_MS = 60000;

// ─── HARDWARE ────────────────────────────────────────────────
Adafruit_BNO055  bno055(55, 0x28);  // sensorID=55, addr=0x28 (ADR pin→GND)
Adafruit_INA260  ina_m2;
Adafruit_INA260  ina_m3;
CytronMD motor1(PWM_DIR, PWM_M1, DIR_M1);
CytronMD motor2(PWM_DIR, PWM_M2, DIR_M2);
CytronMD motor3(PWM_DIR, PWM_M3, DIR_M3);
Encoder  enc1(ENC1_A, ENC1_B);
Encoder  enc2(ENC2_A, ENC2_B);
Encoder  enc3(ENC3_A, ENC3_B);

// ─── BNO055 STATE ────────────────────────────────────────────
static float bno_yaw_deg_raw   = 0.0f;
static float bno_ax_mps2       = 0.0f;
static float bno_ay_mps2       = 0.0f;
static float yaw_zero_offset_f = 0.0f;

// ─── ADXL345 STATE ───────────────────────────────────────────
static int16_t adxl_ax = 0, adxl_ay = 0, adxl_az = 0;
static bool    adxl_ok = false;

// ─── CURRENT SENSING STATE ───────────────────────────────────
static float i2_mA = 0.0f, v2_V = 0.0f, p2_mW = 0.0f;
static float i3_mA = 0.0f, v3_V = 0.0f, p3_mW = 0.0f;
static float i_sum_mA          = 0.0f;
static float i_sum_baseline_mA = 0.0f;
static float i_delta_mA        = 0.0f;

// ─── MOTION STATE ────────────────────────────────────────────
float curr_x_mm = 0.0f, curr_y_mm = 0.0f, curr_theta_deg = 0.0f;
float goal_x_mm = 0.0f, goal_y_mm = 0.0f, goal_theta_deg = 0.0f;
float path_theta_deg = 0.0f, path_dist_mm = 0.0f;
float counts_per_mm;
bool  moving   = false;
long  target2  = 0, target3 = 0;
bool  rotating = false;
float target_yaw_deg = 0.0f;
enum MotionPhase { IDLE = 0, ROTATE_TO_PATH, DRIVE_STRAIGHT, ROTATE_TO_FINAL };
MotionPhase phase = IDLE;

// ─── IMU VELOCITY / SLIP STATE ───────────────────────────────
static float ax_raw = 0.0f, ay_raw = 0.0f;
static float ax_f   = 0.0f, ay_f   = 0.0f;
static float vx_imu = 0.0f, vy_imu = 0.0f;
static float alpha_x_hat = 1.0f, P_x = P0;
static float alpha_y_hat = 1.0f, P_y = P0;
static float alpha_x_use = 1.0f, alpha_y_use = 1.0f;
static unsigned long last_us = 0;
static long  c2_start_straight = 0, c3_start_straight = 0;
static long  dc2_plan = 0, dc3_plan = 0;
static float v1_mps_dbg = 0.0f, v2_mps_dbg = 0.0f, v3_mps_dbg = 0.0f;
static float vx_enc_dbg = 0.0f, vy_enc_dbg = 0.0f, w_enc_dbg = 0.0f;
static float dt_enc_dbg = 0.01f;

// ── Velocity control (v command) ─────────────────────────────
static unsigned long lastVelCmdMs = 0;
static bool          velActive    = false;
const  unsigned long VEL_TIMEOUT_MS = 500;

// ============================================================
//  MATH UTILITIES
// ============================================================
float degNormalize180(float a) {
  while (a >  180.0f) a -= 360.0f;
  while (a < -180.0f) a += 360.0f;
  return a;
}
int wrapToSigned180(int deg) {
  if (deg < 0)   deg += 360;
  if (deg > 180) deg -= 360;
  return deg;
}
inline float clampf(float x, float lo, float hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}
inline float lpf1(float prev, float x, float fc_hz, float dt_s) {
  float RC = 1.0f / (2.0f * 3.1415926f * fc_hz);
  float a  = dt_s / (dt_s + RC);
  return prev + a * (x - prev);
}
// Boost any non-zero PWM to PWM_MOVE_MIN so the robot always starts moving.
// Zero stays zero (respect intentional stops).
inline int deadbandBoost(int pwm) {
  if (pwm == 0) return 0;
  return (pwm > 0) ? max(pwm, PWM_MOVE_MIN) : min(pwm, -PWM_MOVE_MIN);
}
inline float softDeadband(float x, float db) {
  if (x >  db) return x - db;
  if (x < -db) return x + db;
  return 0.0f;
}
float dtFromMicros() {
  unsigned long now_us = micros();
  if (last_us == 0) { last_us = now_us; return 0.01f; }
  unsigned long du = now_us - last_us;
  last_us = now_us;
  return clampf((float)du * 1e-6f, DT_MIN_S, DT_MAX_S);
}

// ============================================================
//  BNO055  (poll — call every loop)
//  Caches heading (0..360°) from Euler orientation report.
//  Caches linear accel (m/s²) for the IMU velocity proxy.
//  NOTE: BNO055 heading increases clockwise (0=North, 90=East).
// ============================================================
void bno055_poll() {
  sensors_event_t event;
  bno055.getEvent(&event);
  bno_yaw_deg_raw = event.orientation.x;  // 0..360°

  imu::Vector<3> la = bno055.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  bno_ax_mps2 = (float)la.x();
  bno_ay_mps2 = (float)la.y();
}

int readYawDeg() {
  // BNO055 is CW-positive (compass). Negate so CCW = positive (math convention).
  float yaw = bno_yaw_deg_raw - yaw_zero_offset_f;
  return wrapToSigned180((int)roundf(-yaw));
}
void zeroYawToCurrent() {
  yaw_zero_offset_f = bno_yaw_deg_raw;
}

// ============================================================
//  ADXL345  (raw Wire reads, data-only)
//  1 LSB ≈ 3.9 mg; Python scaling: raw / 256.0 ≈ g
// ============================================================
static void adxl345_write_reg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}
bool adxl345_begin() {
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(0x00);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_I2C_ADDR, (uint8_t)1);
  if (!Wire.available() || Wire.read() != 0xE5) return false;
  adxl345_write_reg(0x2C, 0x0A); // BW_RATE:    100 Hz output
  adxl345_write_reg(0x31, 0x00); // DATA_FORMAT: ±2 g, 10-bit
  adxl345_write_reg(0x2D, 0x08); // POWER_CTL:  measure mode
  return true;
}
void adxl345_read(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(0x32);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_I2C_ADDR, (uint8_t)6);
  if (Wire.available() >= 6) {
    uint8_t buf[6];
    for (uint8_t i = 0; i < 6; i++) buf[i] = Wire.read();
    ax = (int16_t)((uint16_t)(buf[1] << 8) | buf[0]);
    ay = (int16_t)((uint16_t)(buf[3] << 8) | buf[2]);
    az = (int16_t)((uint16_t)(buf[5] << 8) | buf[4]);
  }
}

// ============================================================
//  CURRENT SENSING  (INA260 × 2)
// ============================================================
void readCurrents() {
  i2_mA = ina_m2.readCurrent();
  v2_V  = ina_m2.readBusVoltage() * 1e-3f;
  p2_mW = ina_m2.readPower();
  i3_mA = ina_m3.readCurrent();
  v3_V  = ina_m3.readBusVoltage() * 1e-3f;
  p3_mW = ina_m3.readPower();
  float dt_approx   = clampf(dt_enc_dbg, DT_MIN_S, DT_MAX_S);
  i_sum_mA          = i2_mA + i3_mA;
  i_sum_baseline_mA = lpf1(i_sum_baseline_mA, i_sum_mA,
                            I_BASELINE_LPF_FC_HZ, dt_approx);
  i_delta_mA        = i_sum_mA - i_sum_baseline_mA;
}

// ============================================================
//  MOTOR HELPERS
// ============================================================
void stopAll() {
  motor1.setSpeed(0);
  motor2.setSpeed(0);
  motor3.setSpeed(0);
}
void applyRotationPWM(int pwm) {
  motor1.setSpeed(pwm);
  motor2.setSpeed(pwm);
  motor3.setSpeed(pwm);
}
void beginRotateTo(float abs_yaw_deg) {
  stopAll();
  moving   = false;
  rotating = true;
  target_yaw_deg = degNormalize180(abs_yaw_deg);
  vx_imu = vy_imu = 0.0f;
}
void updateYawControl(int yaw_deg) {
  if (!rotating) return;
  float err = target_yaw_deg - (float)yaw_deg;
  if (err >  180.0f) err -= 360.0f;
  if (err < -180.0f) err += 360.0f;
  if (fabs(err) < YAW_DEADBAND) { stopAll(); rotating = false; return; }
  applyRotationPWM(deadbandBoost((int)clampf(YAW_KP * err, -PWM_YAW_MAX, PWM_YAW_MAX)));
}

// ============================================================
//  IMU VELOCITY PROXY
// ============================================================
void imuVelocityProxyUpdate(bool zupt) {
  float dt = dtFromMicros();
  float ax = softDeadband(bno_ax_mps2, ACC_DEADBAND_MPS2);
  float ay = softDeadband(bno_ay_mps2, ACC_DEADBAND_MPS2);
  ax_raw = bno_ax_mps2;
  ay_raw = bno_ay_mps2;
  ax_f    = lpf1(ax_f, ax, ACC_LPF_FC_HZ, dt);
  ay_f    = lpf1(ay_f, ay, ACC_LPF_FC_HZ, dt);
  vx_imu += ax_f * dt;
  vy_imu += ay_f * dt;
  float leak = clampf(VEL_LEAK_PER_S * dt, 0.0f, 0.30f);
  vx_imu *= (1.0f - leak);
  vy_imu *= (1.0f - leak);
  if (zupt) { vx_imu = vy_imu = 0.0f; }
}

// ============================================================
//  WHEEL SPEEDS + FORWARD KINEMATICS
// ============================================================
void readWheelSpeeds_mps(float &v1, float &v2, float &v3, float &dt) {
  static long last_c1 = 0, last_c2 = 0, last_c3 = 0;
  static unsigned long last_t = 0;
  unsigned long now = micros();
  if (last_t == 0) {
    last_t = now;
    last_c1 = enc1.read(); last_c2 = enc2.read(); last_c3 = enc3.read();
    dt = 0.01f; v1 = v2 = v3 = 0.0f;
    return;
  }
  unsigned long du = now - last_t;
  dt = clampf((float)du * 1e-6f, DT_MIN_S, DT_MAX_S);
  long c1 = enc1.read(), c2 = enc2.read(), c3 = enc3.read();
  long dc1 = -(c1 - last_c1), dc2 = -(c2 - last_c2), dc3 = -(c3 - last_c3);
  last_c1 = c1; last_c2 = c2; last_c3 = c3; last_t = now;
  v1 = ((float)dc1 / counts_per_mm / dt) * 1e-3f;
  v2 = ((float)dc2 / counts_per_mm / dt) * 1e-3f;
  v3 = ((float)dc3 / counts_per_mm / dt) * 1e-3f;
}
void forwardKinematics120(float v1, float v2, float v3,
                          float &vx, float &vy, float &w_rad_s) {
  const float R   = WHEEL_RADIUS_MM * 1e-3f;
  const float L   = ROBOT_RADIUS_MM * 1e-3f;
  const float rt3 = 1.7320508f;
  float u1 = v1/R, u2 = v2/R, u3 = v3/R;
  vx      = (R / 3.0f)       * (-u2 + u3 - 2.0f * u1);
  vy      = (R / rt3)         * (u3 + u2);
  w_rad_s = (R / (3.0f*L))   * (u1 + u2 + u3);
}

// ============================================================
//  RLS SLIP ESTIMATOR
// ============================================================
void rlsUpdateScalar(float y, float phi, float &theta, float &P) {
  float denom = RLS_ETA + phi * phi * P;
  if (denom < 1e-6f) return;
  float K   = (P * phi) / denom;
  float err = y - phi * theta;
  theta += K * err;
  P = (1.0f / RLS_ETA) * (P - K * phi * P);
  theta = clampf(theta, ALPHA_MIN, ALPHA_MAX);
  P     = clampf(P, 1e-4f, 1e6f);
}
void slipEstimatorUpdate(bool enable) {
  if (!enable) return;
  bool zupt = (fabs(vx_enc_dbg) < 0.02f && fabs(vy_enc_dbg) < 0.02f);
  imuVelocityProxyUpdate(zupt);
  float v1, v2, v3, dt;
  readWheelSpeeds_mps(v1, v2, v3, dt);
  float vx_enc, vy_enc, w_enc;
  forwardKinematics120(v1, v2, v3, vx_enc, vy_enc, w_enc);
  v1_mps_dbg = v1; v2_mps_dbg = v2; v3_mps_dbg = v3;
  vx_enc_dbg = vx_enc; vy_enc_dbg = vy_enc; w_enc_dbg = w_enc;
  dt_enc_dbg = dt;
  if (fabs(vx_enc) > RLS_VMIN_MPS)
    rlsUpdateScalar(fabs(vx_imu), fabs(vx_enc), alpha_x_hat, P_x);
  if (fabs(vy_enc) > RLS_VMIN_MPS)
    rlsUpdateScalar(fabs(vy_imu), fabs(vy_enc), alpha_y_hat, P_y);
  alpha_x_use = lpf1(alpha_x_use, alpha_x_hat, ALPHA_LPF_FC_HZ, dt);
  alpha_y_use = lpf1(alpha_y_use, alpha_y_hat, ALPHA_LPF_FC_HZ, dt);
  alpha_x_use = clampf(alpha_x_use, ALPHA_MIN, ALPHA_MAX);
  alpha_y_use = clampf(alpha_y_use, ALPHA_MIN, ALPHA_MAX);
}

// ============================================================
//  MOTION EXECUTION
// ============================================================
void startMoveAlongWheel1(float distance_mm) {
  rotating = false;
  stopAll();
  vx_imu = vy_imu = 0.0f;
  ax_f = ay_f = 0.0f;
  alpha_x_hat = alpha_y_hat = 1.0f;
  alpha_x_use = alpha_y_use = 1.0f;
  P_x = P_y = P0;
  c2_start_straight = -enc2.read();
  c3_start_straight = -enc3.read();
  const float factor = sqrt(3.0f) / 2.0f;
  float s2_mm = -factor * distance_mm;
  float s3_mm = +factor * distance_mm;
  dc2_plan = (long)lrintf(s2_mm * counts_per_mm);
  dc3_plan = (long)lrintf(s3_mm * counts_per_mm);
  target2  = c2_start_straight + dc2_plan;
  target3  = c3_start_straight + dc3_plan;
  moving   = true;
}
void updatePositionControl() {
  if (!moving) return;
  // Run estimator for telemetry only — alpha is NOT applied to targets.
  // Slip correction is handled by the ZED pose feedback loop on the Jetson.
  slipEstimatorUpdate(true);
  target2 = c2_start_straight + dc2_plan;
  target3 = c3_start_straight + dc3_plan;
  long c2 = -enc2.read(), c3 = -enc3.read();
  long e2 = target2 - c2, e3 = target3 - c3;
  if (labs(e2) < COUNT_TOL || labs(e3) < COUNT_TOL) {
    stopAll();
    moving = false;
    vx_imu = vy_imu = 0.0f;
    return;
  }
  int pwm2 = deadbandBoost((int)constrain(KP_POS * (float)e2, -PWM_POS_MAX, PWM_POS_MAX));
  int pwm3 = deadbandBoost((int)constrain(KP_POS * (float)e3, -PWM_POS_MAX, PWM_POS_MAX));
  motor1.setSpeed(0);
  motor2.setSpeed(-pwm2);
  motor3.setSpeed(-pwm3);
}
void planGoto(float gx_units, float gy_units, float gth_deg) {
  goal_x_mm = gx_units * GRID_UNIT_MM;
  goal_y_mm = gy_units * GRID_UNIT_MM;
  goal_theta_deg = gth_deg;
  float dx = goal_x_mm - curr_x_mm, dy = goal_y_mm - curr_y_mm;
  path_dist_mm = sqrt(dx*dx + dy*dy);
  if (path_dist_mm < 1.0f) {
    path_theta_deg = curr_theta_deg;
  } else {
    path_theta_deg = degNormalize180(atan2f(dy, dx) * 180.0f / PI);
  }
  if (path_dist_mm < 1.0f) {
    beginRotateTo(goal_theta_deg);
    phase = ROTATE_TO_FINAL;
    return;
  }
  beginRotateTo(path_theta_deg);
  phase = ROTATE_TO_PATH;
}

// ============================================================
//  SERIAL LINE READER
// ============================================================
bool readLine(String &out) {
  static String buf = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    // Serial.write(c); // echo removed — reduces latency
    if (c == '\n' || c == '\r') {
      out = buf; buf = ""; out.trim();
      return (out.length() > 0);
    } else {
      buf += c;
      if (buf.length() > 120) buf = "";
    }
  }
  return false;
}

// ============================================================
//  TELEMETRY  (ASCII CSV prefixed "TEL,")
//  Jetson parser splits on ',' and checks first token == "TEL"
// ============================================================
void logMaybePrint() {
  if (!LOG_ON) return;
  unsigned long nowMs = millis();
  if (nowMs - lastLogMs < LOG_DT_MS) return;
  lastLogMs = nowMs;

  Serial.print("TEL,");
  Serial.print(nowMs);          Serial.print(",");
  Serial.print(LOG_LABEL);      Serial.print(",");
  Serial.print(-enc1.read());   Serial.print(",");
  Serial.print(-enc2.read());   Serial.print(",");
  Serial.print(-enc3.read());   Serial.print(",");
  Serial.print(vx_enc_dbg, 4);  Serial.print(",");
  Serial.print(ax_raw, 4);      Serial.print(",");
  Serial.print(ax_f,    4);     Serial.print(",");
  Serial.print(vx_imu,  4);     Serial.print(",");
  Serial.print(alpha_x_hat, 4); Serial.print(",");
  Serial.print(alpha_x_use, 4); Serial.print(",");
  Serial.print(i_sum_mA,    2); Serial.print(",");
  Serial.print(i_delta_mA,  2); Serial.print(",");
  Serial.print(adxl_ax);        Serial.print(",");
  Serial.print(adxl_ay);        Serial.print(",");
  Serial.println(adxl_az);
}

// ============================================================
//  COMMAND HANDLER
// ============================================================
void handleCommands() {
  String line;
  if (!readLine(line)) return;

  // Auto-stop logging on any non-log command
  if (LOG_ON && !line.startsWith("log")) {
    LOG_ON = false; lastLogMs = 0;
    Serial.println("\r\n[log auto-stopped]");
  }

  // ── log 1 <label>  /  log 0 ──────────────────────────────
  if (line.startsWith("log")) {
    String rest = line.substring(3); rest.trim();
    if (rest.length() == 0) {
      Serial.println("\r\nUsage: log 1 label  |  log 0"); return;
    }
    int sp = rest.indexOf(' ');
    String tokOn = (sp < 0) ? rest : rest.substring(0, sp); tokOn.trim();
    int on = tokOn.toInt();
    if (on == 0) {
      LOG_ON = false; Serial.println("\r\n[log off]"); return;
    }
    LOG_ON = true; lastLogMs = 0;
    if (sp >= 0) {
      String lab = rest.substring(sp + 1); lab.trim();
      if (lab.length() > 0) lab.toCharArray(LOG_LABEL, sizeof(LOG_LABEL));
    }
    Serial.print("\r\n[log on] label="); Serial.println(LOG_LABEL);
    return;
  }

  // ── g x y theta ──────────────────────────────────────────
  if (line.charAt(0) == 'g') {
    String rest = line.substring(1); rest.trim();
    int s1 = rest.indexOf(' ');
    if (s1 < 0) { Serial.println("\r\nUsage: g x y theta"); return; }
    String tok1 = rest.substring(0, s1); tok1.trim();
    String rest2 = rest.substring(s1 + 1); rest2.trim();
    int s2 = rest2.indexOf(' ');
    if (s2 < 0) { Serial.println("\r\nUsage: g x y theta"); return; }
    String tok2 = rest2.substring(0, s2); tok2.trim();
    String tok3 = rest2.substring(s2 + 1); tok3.trim();
    planGoto(tok1.toFloat(), tok2.toFloat(), tok3.toFloat());
    LOG_ON = true; lastLogMs = 0;
    strncpy(LOG_LABEL, "goto", sizeof(LOG_LABEL) - 1);
    autoGotoLog = true; autoLogStartMs = millis();
    Serial.println("\r\n[goto accepted]");
    return;
  }

  // ── stop ─────────────────────────────────────────────────
  if (line == "stop") {
    phase = IDLE; moving = false; rotating = false;
    stopAll(); imuVelocityProxyUpdate(true);
    Serial.println("\r\n[stopped]");
    return;
  }

  // ── z (zero / reset) ─────────────────────────────────────
  if (line == "z") {
    zeroYawToCurrent();
    curr_x_mm = curr_y_mm = curr_theta_deg = 0.0f;
    phase = IDLE; moving = rotating = false;
    stopAll(); vx_imu = vy_imu = 0.0f;
    Serial.println("\r\n[reset]");
    return;
  }

  // ── s (status) ───────────────────────────────────────────
  if (line == "s") {
    Serial.print("\r\n[status] phase="); Serial.print((int)phase);
    Serial.print(" yaw=");    Serial.print(readYawDeg());
    Serial.print(" ax=");     Serial.print(alpha_x_use, 3);
    Serial.print(" ay=");     Serial.print(alpha_y_use, 3);
    Serial.print(" I_sum=");  Serial.print(i_sum_mA,    1);
    Serial.print(" I_dlt=");  Serial.print(i_delta_mA,  1);
    Serial.print(" adxl=");   Serial.print(adxl_ax); Serial.print(",");
                               Serial.print(adxl_ay); Serial.print(",");
                               Serial.println(adxl_az);
    return;
  }

  // ── y (quick yaw) ─────────────────────────────────────────
  if (line == "y") {
    Serial.print("\r\n[yaw] "); Serial.println(readYawDeg());
    return;
  }

  // ── spd <0-100>  (traction-based speed cap from Jetson) ─────────────
  if (line.startsWith("spd")) {
    String rest = line.substring(3); rest.trim();
    int pct = constrain(rest.toInt(), 0, 100);
    PWM_POS_MAX = map(pct, 0, 100, 0, 100);
    PWM_YAW_MAX = map(pct, 0, 100, 0, 50);
    Serial.print("\r\n[speed] "); Serial.println(pct);
    return;
  }

  // ── v fwd turn  (velocity control — Jetson closed-loop) ──────────
  if (line.charAt(0) == 'v') {
    String rest = line.substring(1); rest.trim();
    int sp = rest.indexOf(' ');
    String tok1 = (sp < 0) ? rest : rest.substring(0, sp); tok1.trim();
    String tok2 = (sp < 0) ? String("0") : rest.substring(sp + 1); tok2.trim();
    int fwd  = constrain(tok1.toInt(), -100, 100);
    int turn = constrain(tok2.toInt(), -100, 100);
    // Cancel any ongoing g-command motion
    phase = IDLE; moving = false; rotating = false;
    // Forward: inverted PWM on motors 2 & 3
    // Turn:    same PWM on all motors
    int m1 =  -turn;
    int m2 =  fwd - turn;
    int m3 = -fwd - turn;
    motor1.setSpeed(deadbandBoost(constrain(m1, -100, 100)));
    motor2.setSpeed(deadbandBoost(constrain(m2, -100, 100)));
    motor3.setSpeed(deadbandBoost(constrain(m3, -100, 100)));
    lastVelCmdMs = millis();
    velActive    = true;
    return;
  }

  Serial.println("\r\nCommands: g x y th | v fwd turn | stop | z | s | y | spd 0-100 | log 1 label | log 0");
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  Wire.begin();

  Serial.println("\n=== SIR 3WD base (USB serial direct) ===");

  // --- BNO055 at 0x28 ---
  Serial.println("Starting BNO055...");
  if (!bno055.begin()) {
    Serial.println("ERROR: BNO055 not detected at 0x28!"); while (1);
  }
  bno055.setExtCrystalUse(true);
  delay(500);
  for (int i = 0; i < 50; i++) { bno055_poll(); delay(5); }
  zeroYawToCurrent();
  Serial.print("BNO055 OK. Yaw zeroed at ");
  Serial.print(bno_yaw_deg_raw, 2); Serial.println(" deg");

  // --- INA260 ---
  Serial.println("Starting INA260...");
  if (!ina_m3.begin(0x40)) { Serial.println("ERROR: INA260 M3 (0x40) not found!"); while (1); }
  if (!ina_m2.begin(0x41)) { Serial.println("ERROR: INA260 M2 (0x41) not found!"); while (1); }
  Serial.println("INA260 OK.");

  // --- ADXL345 ---
  adxl_ok = adxl345_begin();
  Serial.println(adxl_ok ? "ADXL345 OK (0x53, ±2g, 100 Hz)."
                          : "ADXL345 not detected — TEL adxl fields will be 0.");

  // --- Encoders ---
  pinMode(ENC1_A, INPUT_PULLUP); pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC2_A, INPUT_PULLUP); pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC3_A, INPUT_PULLUP); pinMode(ENC3_B, INPUT_PULLUP);

  counts_per_mm = COUNTS_PER_REV / (2.0f * PI * WHEEL_RADIUS_MM);
  stopAll();
  phase = IDLE; moving = false; rotating = false; last_us = 0;

  Serial.print("Ready. counts_per_mm="); Serial.println(counts_per_mm, 3);
  Serial.println("Commands: g x y th | stop | z | s | y | log 1 label | log 0");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  // 1. Poll sensors (non-blocking)
  bno055_poll();
  if (adxl_ok) adxl345_read(adxl_ax, adxl_ay, adxl_az);

  // 2. Handle incoming commands from Jetson
  // Drain ALL queued commands so the latest 'v' always wins
  do { handleCommands(); } while (Serial.available());

  // 3. Slip estimator + current sensing
  slipEstimatorUpdate(true);
  readCurrents();

  // 4. Velocity watchdog — stop if Jetson hasn't sent 'v' recently
  if (velActive && (millis() - lastVelCmdMs > VEL_TIMEOUT_MS)) {
    velActive = false;
    stopAll();
  }

  // 5. Motor control (g-command path only)
  int yaw = readYawDeg();
  updateYawControl(yaw);
  updatePositionControl();

  // 5. Phase state machine
  switch (phase) {
    case ROTATE_TO_PATH:
      if (!rotating) {
        startMoveAlongWheel1(path_dist_mm);
        phase = DRIVE_STRAIGHT;
      }
      break;
    case DRIVE_STRAIGHT:
      if (!moving) {
        curr_x_mm = goal_x_mm; curr_y_mm = goal_y_mm;
        beginRotateTo(goal_theta_deg);
        phase = ROTATE_TO_FINAL;
      }
      break;
    case ROTATE_TO_FINAL:
      if (!rotating) {
        curr_theta_deg = degNormalize180(goal_theta_deg);
        phase = IDLE;
        Serial.println("[done]");
      }
      break;
    case IDLE:
    default:
      imuVelocityProxyUpdate(true);
      break;
  }

  // 6. Emit telemetry if logging active
  logMaybePrint();

  // 7. Auto-stop goto telemetry after timeout
  if (autoGotoLog && millis() - autoLogStartMs > AUTO_LOG_DUR_MS) {
    LOG_ON = false; autoGotoLog = false;
    Serial.println("[auto log stopped]");
  }
}
