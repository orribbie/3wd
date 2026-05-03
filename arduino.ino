// ============================================================
//  arduino_log.ino  —  Rev 2: BNO08x + ADXL345 telemetry
//  Changes from Rev 1:
//    • BNO055  →  BNO08x  (Adafruit_BNO08x, I2C 0x4B)
//      - GAME_ROTATION_VECTOR for yaw (quaternion → Euler)
//      - LINEAR_ACCELERATION for ax/ay
//      - bno08x_poll() called every loop; yaw/accel cached
//    • ADXL345 added at I2C 0x53 (data-only, no control)
//      - adxl345_begin() / adxl345_read() via raw Wire
//      - raw 10-bit counts added to TelemetryPacket
//    • TelemetryPacket switched floats → int16 (×scale)
//      to match Python parser "<I16siii11h" (54 bytes)
//    • i_sum_mA / i_delta_mA computed in readCurrents()
// ============================================================

#include <Adafruit_BNO08x.h>
#include <Adafruit_INA260.h>
#include <CytronMotorDriver.h>
#include <Encoder.h>
#include <Wire.h>
#include <math.h>

// ─── UART LINK TO XIAO ──────────────────────────────────────
#define ESP_SERIAL Serial3
static const uint32_t ESP_BAUD = 115200;

// ─── UART MUX FRAMING ───────────────────────────────────────
static const uint8_t MUX_MAGIC0 = 0xAA;
static const uint8_t MUX_MAGIC1 = 0x55;
enum MuxType : uint8_t { MUX_CMD = 1, MUX_TEL = 2, MUX_RSP = 3 };

static const uint16_t CMD_RX_CAP = 512;
static volatile uint16_t cmdRxHead = 0, cmdRxTail = 0;
static uint8_t cmdRxBuf[CMD_RX_CAP];
static inline void cmdRxPush(uint8_t b) {
  uint16_t n = (cmdRxHead + 1) % CMD_RX_CAP;
  if (n == cmdRxTail) return;
  cmdRxBuf[cmdRxHead] = b;
  cmdRxHead = n;
}
static inline int cmdRxPop() {
  if (cmdRxHead == cmdRxTail) return -1;
  uint8_t b = cmdRxBuf[cmdRxTail];
  cmdRxTail = (cmdRxTail + 1) % CMD_RX_CAP;
  return b;
}
static inline int cmdRxCount() {
  if (cmdRxHead >= cmdRxTail) return (int)(cmdRxHead - cmdRxTail);
  return (int)(CMD_RX_CAP - (cmdRxTail - cmdRxHead));
}

enum RxState { RX_WAIT_AA, RX_WAIT_55, RX_TYPE, RX_LEN0, RX_LEN1, RX_SEQ, RX_PAYLOAD };
static RxState  rxState = RX_WAIT_AA;
static uint8_t  rxType  = 0;
static uint16_t rxLen   = 0;
static uint8_t  rxSeq   = 0;
static uint16_t rxGot   = 0;

static void muxPoll() {
  while (ESP_SERIAL.available()) {
    uint8_t c = (uint8_t)ESP_SERIAL.read();
    switch (rxState) {
      case RX_WAIT_AA: rxState = (c == MUX_MAGIC0) ? RX_WAIT_55 : RX_WAIT_AA; break;
      case RX_WAIT_55: rxState = (c == MUX_MAGIC1) ? RX_TYPE    : RX_WAIT_AA; break;
      case RX_TYPE:    rxType  = c; rxState = RX_LEN0; break;
      case RX_LEN0:    rxLen   = c; rxState = RX_LEN1; break;
      case RX_LEN1:    rxLen  |= ((uint16_t)c << 8); rxState = RX_SEQ; break;
      case RX_SEQ:     rxSeq   = c; rxGot = 0; rxState = RX_PAYLOAD; break;
      case RX_PAYLOAD:
        if (rxType == MUX_CMD) cmdRxPush(c);
        if (++rxGot >= rxLen)  rxState = RX_WAIT_AA;
        break;
    }
  }
}
static uint8_t txSeq = 0;
static void muxSendFrame(uint8_t type, const uint8_t *payload, uint16_t len) {
  uint8_t seq = txSeq++;
  ESP_SERIAL.write(MUX_MAGIC0);
  ESP_SERIAL.write(MUX_MAGIC1);
  ESP_SERIAL.write(type);
  ESP_SERIAL.write((uint8_t)(len & 0xFF));
  ESP_SERIAL.write((uint8_t)((len >> 8) & 0xFF));
  ESP_SERIAL.write(seq);
  if (len && payload) ESP_SERIAL.write(payload, len);
}

class CmdStream : public Stream {
public:
  int    available() override { return cmdRxCount(); }
  int    read()      override { return cmdRxPop(); }
  int    peek()      override { return (cmdRxHead == cmdRxTail) ? -1 : cmdRxBuf[cmdRxTail]; }
  void   flush()     override {}
  size_t write(uint8_t b) override { muxSendFrame(MUX_RSP, &b, 1); return 1; }
};
CmdStream WIFI_PORT;
#define BT_SERIAL WIFI_PORT

// ─── CONFIG ─────────────────────────────────────────────────
const float GRID_UNIT_MM     = 100.0f;
const float WHEEL_RADIUS_MM  = 60.0f;
const int   ENCODER_PPR_MOTOR = 4;
const float GEAR_RATIO       = 41.6f;
const float COUNTS_PER_REV   = ENCODER_PPR_MOTOR * GEAR_RATIO * 16.0f;
const float KP_POS           = 0.2f;
const int   PWM_POS_MAX      = 100;
const long  COUNT_TOL        = 20;
const float YAW_KP           = 3.0f;
const float YAW_DEADBAND     = 3.0f;
const int   PWM_YAW_MAX      = 50;

const uint8_t PWM_M1 = 6,  DIR_M1 = 7;
const uint8_t PWM_M2 = 4,  DIR_M2 = 5;
const uint8_t PWM_M3 = 8,  DIR_M3 = 10;
const int ENC1_A = 13, ENC1_B = 3;
const int ENC2_A = 2,  ENC2_B = 9;
const int ENC3_A = 18, ENC3_B = 11;

// Sensor addresses
const uint8_t BNO08X_ADDR     = 0x4B;   // BNO08x I2C address (ADR pin high)
const uint8_t ADXL345_I2C_ADDR = 0x53;  // ADXL345 I2C address (SDO low)

const unsigned long TELEMETRY_DT_MS  = 200;
const float ROBOT_RADIUS_MM          = 150.0f;
const float DT_MIN_S                 = 0.002f;
const float DT_MAX_S                 = 0.05f;
const float ACC_DEADBAND_MPS2        = 0.12f;
const float ACC_LPF_FC_HZ            = 6.0f;
const float VEL_LEAK_PER_S           = 0.69f;
const float RLS_ETA                  = 0.99f;
const float RLS_VMIN_MPS             = 0.03f;
const float ALPHA_MIN                = 0.93f;
const float ALPHA_MAX                = 1.00f;
const float P0                       = 50.0f;
const float ALPHA_LPF_FC_HZ          = 1.5f;
const float I_BASELINE_LPF_FC_HZ     = 0.5f;  // LPF for I_delta baseline

static bool             LOG_ON            = false;
static char             LOG_LABEL[16]     = "none";
const unsigned long     LOG_DT_MS         = 40;
static unsigned long    lastLogMs         = 0;
static bool             logHeaderPrinted  = false;
static Stream          *LOG_PORT          = &Serial;
static bool             autoGotoLogging   = false;
static unsigned long    autoLogStartMs    = 0;
const unsigned long     AUTO_LOG_DURATION_MS = 60000;

// ─── HARDWARE OBJECTS ────────────────────────────────────────
Adafruit_BNO08x  bno08x;
Adafruit_INA260  ina_m2;
Adafruit_INA260  ina_m3;
CytronMD motor1(PWM_DIR, PWM_M1, DIR_M1);
CytronMD motor2(PWM_DIR, PWM_M2, DIR_M2);
CytronMD motor3(PWM_DIR, PWM_M3, DIR_M3);
Encoder  enc1(ENC1_A, ENC1_B);
Encoder  enc2(ENC2_A, ENC2_B);
Encoder  enc3(ENC3_A, ENC3_B);

// ─── BNO08x CACHED STATE ─────────────────────────────────────
static float bno_yaw_deg_raw = 0.0f;   // yaw from game rotation vector (degrees)
static float bno_ax_mps2     = 0.0f;   // linear accel X (m/s²)
static float bno_ay_mps2     = 0.0f;   // linear accel Y (m/s²)
static float yaw_zero_offset_f = 0.0f; // float zeroing offset

// ─── ADXL345 RAW READINGS ────────────────────────────────────
static int16_t adxl_ax = 0, adxl_ay = 0, adxl_az = 0;
static bool    adxl_ok = false;

// ─── CURRENT STATE ───────────────────────────────────────────
static float i2_mA = 0.0f, v2_V = 0.0f, p2_mW = 0.0f;
static float i3_mA = 0.0f, v3_V = 0.0f, p3_mW = 0.0f;
static float i_sum_mA          = 0.0f;  // I2 + I3
static float i_sum_baseline_mA = 0.0f;  // slow LPF of i_sum
static float i_delta_mA        = 0.0f;  // i_sum - baseline (spike signal)

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

// ─── IMU VELOCITY PROXY STATE ────────────────────────────────
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

// ============================================================
//  UTILITIES
// ============================================================
float degNormalize180(float a) {
  while (a >  180.0f) a -= 360.0f;
  while (a < -180.0f) a += 360.0f;
  return a;
}
int wrapToSigned180(int deg) {
  if (deg < 0)    deg += 360;
  if (deg > 180)  deg -= 360;
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
//  BNO08x — ASYNC POLL (call every loop)
//  Caches yaw (degrees) and linear accel (m/s²) from sensor events.
// ============================================================
void bno08x_poll() {
  sh2_SensorValue_t val;
  while (bno08x.getSensorEvent(&val)) {
    switch (val.sensorId) {
      case SH2_GAME_ROTATION_VECTOR: {
        // Quaternion: real=w, i=x, j=y, k=z
        float qw = val.un.gameRotationVector.real;
        float qx = val.un.gameRotationVector.i;
        float qy = val.un.gameRotationVector.j;
        float qz = val.un.gameRotationVector.k;
        // Yaw (rotation about Z) in degrees
        float yaw_rad = atan2f(2.0f * (qw * qz + qx * qy),
                               1.0f - 2.0f * (qy * qy + qz * qz));
        bno_yaw_deg_raw = yaw_rad * (180.0f / PI);
        break;
      }
      case SH2_LINEAR_ACCELERATION:
        bno_ax_mps2 = val.un.linearAcceleration.x;
        bno_ay_mps2 = val.un.linearAcceleration.y;
        break;
      default:
        break;
    }
  }
}

int readYawDeg() {
  float yaw = bno_yaw_deg_raw - yaw_zero_offset_f;
  return wrapToSigned180((int)roundf(yaw));
}
void zeroYawToCurrent() {
  yaw_zero_offset_f = bno_yaw_deg_raw;
}

// ============================================================
//  ADXL345 — RAW WIRE READS (data-only, no control use)
//  Stores int16 counts: 1 LSB ≈ 3.9 mg at ±2 g full-res.
//  Python scaling: raw / 256.0  ≈  acceleration in g.
// ============================================================
static void adxl345_write_reg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}
bool adxl345_begin() {
  // Verify DEVID register (should read 0xE5)
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(0x00);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_I2C_ADDR, (uint8_t)1);
  if (!Wire.available() || Wire.read() != 0xE5) return false;
  adxl345_write_reg(0x2C, 0x0A); // BW_RATE   : 100 Hz output
  adxl345_write_reg(0x31, 0x00); // DATA_FORMAT: ±2 g, 10-bit, right-justified
  adxl345_write_reg(0x2D, 0x08); // POWER_CTL : enter measure mode
  return true;
}
void adxl345_read(int16_t &ax, int16_t &ay, int16_t &az) {
  Wire.beginTransmission(ADXL345_I2C_ADDR);
  Wire.write(0x32);              // DATAX0 — burst-read 6 bytes
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
//  CURRENT SENSING
//  Computes i_sum = I2+I3 and i_delta = i_sum - slow-LPF baseline.
// ============================================================
void readCurrents() {
  i2_mA = ina_m2.readCurrent();
  v2_V  = ina_m2.readBusVoltage() * 1e-3f;
  p2_mW = ina_m2.readPower();
  i3_mA = ina_m3.readCurrent();
  v3_V  = ina_m3.readBusVoltage() * 1e-3f;
  p3_mW = ina_m3.readPower();
  // I_sum and I_delta (use dt_enc_dbg as approximate loop dt)
  float dt_approx       = clampf(dt_enc_dbg, DT_MIN_S, DT_MAX_S);
  i_sum_mA              = i2_mA + i3_mA;
  i_sum_baseline_mA     = lpf1(i_sum_baseline_mA, i_sum_mA,
                                I_BASELINE_LPF_FC_HZ, dt_approx);
  i_delta_mA            = i_sum_mA - i_sum_baseline_mA;
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
  motor1.setSpeed(-pwm);
  motor2.setSpeed(-pwm);
  motor3.setSpeed(-pwm);
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
  applyRotationPWM((int)clampf(YAW_KP * err, -PWM_YAW_MAX, PWM_YAW_MAX));
}

// ============================================================
//  IMU VELOCITY PROXY  (unchanged except source of ax/ay)
// ============================================================
void imuVelocityProxyUpdate(bool zupt) {
  float dt = dtFromMicros();
  // Use BNO08x linear acceleration (cached by bno08x_poll)
  float ax = bno_ax_mps2;
  float ay = bno_ay_mps2;
  ax_raw = ax;
  ay_raw = ay;
  ax = softDeadband(ax, ACC_DEADBAND_MPS2);
  ay = softDeadband(ay, ACC_DEADBAND_MPS2);
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
//  WHEEL SPEEDS + FORWARD KINEMATICS  (unchanged)
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
  vx    = (R / 3.0f)       * (-u2 + u3 - 2.0f * u1);
  vy    = (R / rt3)         * (u3 + u2);
  w_rad_s = (R / (3.0f*L)) * (u1 + u2 + u3);
}

// ============================================================
//  RLS SLIP ESTIMATOR  (unchanged)
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
//  MOTION EXECUTION  (unchanged)
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
  slipEstimatorUpdate(true);
  float ax       = clampf(alpha_x_use, ALPHA_MIN, ALPHA_MAX);
  long dc2_scaled = (long)lrintf((float)dc2_plan / ax);
  long dc3_scaled = (long)lrintf((float)dc3_plan / ax);
  target2 = c2_start_straight + dc2_scaled;
  target3 = c3_start_straight + dc3_scaled;
  long c2 = -enc2.read(), c3 = -enc3.read();
  long e2 = target2 - c2, e3 = target3 - c3;
  if (labs(e2) < COUNT_TOL || labs(e3) < COUNT_TOL) {
    stopAll();
    moving   = false;
    vx_imu = vy_imu = 0.0f;
    return;
  }
  float pwm2_f = constrain(KP_POS * (float)e2, -PWM_POS_MAX, PWM_POS_MAX);
  float pwm3_f = constrain(KP_POS * (float)e3, -PWM_POS_MAX, PWM_POS_MAX);
  motor1.setSpeed(0);
  motor2.setSpeed(-(int)pwm2_f);
  motor3.setSpeed(-(int)pwm3_f);
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
//  COMMAND STREAM HELPERS  (unchanged)
// ============================================================
bool readLineFrom(Stream &port, String &out) {
  static String bufUSB = "", bufBT = "";
  String *buf = (&port == (Stream *)&BT_SERIAL) ? &bufBT : &bufUSB;
  while (port.available()) {
    char c = (char)port.read();
    if (&port == (Stream *)&Serial) port.write(c);
    if (c == '\n' || c == '\r') {
      out = *buf; buf->remove(0); out.trim();
      return (out.length() > 0);
    } else {
      *buf += c;
      if (buf->length() > 120) buf->remove(0);
    }
  }
  return false;
}

// ============================================================
//  TELEMETRY PACKET  (Rev 2: int16 scaled + ADXL345)
//
//  Python format string: "<I16siii11h"   (54 bytes total)
//  Scaling (Arduino packs → Python unpacks):
//    vx_enc      ×1000  →  /1000   m/s
//    ax_raw      ×100   →  /100    m/s²
//    ax_f        ×100   →  /100    m/s²
//    vx_imu      ×1000  →  /1000   m/s
//    alpha_x_hat ×1000  →  /1000   (dimensionless)
//    alpha_x_use ×1000  →  /1000   (dimensionless)
//    i_sum       ×10    →  /10     mA
//    i_delta     ×10    →  /10     mA
//    adxl_ax     raw counts → /256.0  ≈ g  (1 LSB ≈ 3.9 mg)
//    adxl_ay     raw counts → /256.0  ≈ g
//    adxl_az     raw counts → /256.0  ≈ g
// ============================================================
#pragma pack(push, 1)
struct TelemetryPacket {
  uint32_t t_ms;
  char     label[16];
  int32_t  enc1, enc2, enc3;
  int16_t  vx_enc;        // m/s   × 1000
  int16_t  ax_raw;        // m/s²  × 100
  int16_t  ax_f;          // m/s²  × 100
  int16_t  vx_imu;        // m/s   × 1000
  int16_t  alpha_x_hat;   // ×1000
  int16_t  alpha_x_use;   // ×1000
  int16_t  i_sum;         // mA    × 10
  int16_t  i_delta;       // mA    × 10
  int16_t  adxl_ax;       // raw ADXL counts
  int16_t  adxl_ay;
  int16_t  adxl_az;
};
#pragma pack(pop)
// Compile-time size guard: expected 4+16+12+22 = 54 bytes
static_assert(sizeof(TelemetryPacket) == 54, "TelemetryPacket size mismatch — check packing");

// Helper: clamp float to int16 after scaling
static inline int16_t f2i16(float val, float scale) {
  long v = lrintf(val * scale);
  if (v >  32767) v =  32767;
  if (v < -32768) v = -32768;
  return (int16_t)v;
}

void logMaybePrint() {
  if (!LOG_ON) return;
  unsigned long nowMs = millis();
  if (nowMs - lastLogMs < LOG_DT_MS) return;
  lastLogMs = nowMs;

  TelemetryPacket p;
  memset(&p, 0, sizeof(p));
  p.t_ms = nowMs;
  strncpy(p.label, LOG_LABEL, sizeof(p.label) - 1);
  p.enc1 = (int32_t)(-enc1.read());
  p.enc2 = (int32_t)(-enc2.read());
  p.enc3 = (int32_t)(-enc3.read());
  p.vx_enc       = f2i16(vx_enc_dbg,  1000.0f);
  p.ax_raw       = f2i16(ax_raw,       100.0f);
  p.ax_f         = f2i16(ax_f,         100.0f);
  p.vx_imu       = f2i16(vx_imu,      1000.0f);
  p.alpha_x_hat  = f2i16(alpha_x_hat, 1000.0f);
  p.alpha_x_use  = f2i16(alpha_x_use, 1000.0f);
  p.i_sum        = f2i16(i_sum_mA,      10.0f);
  p.i_delta      = f2i16(i_delta_mA,    10.0f);
  p.adxl_ax      = adxl_ax;
  p.adxl_ay      = adxl_ay;
  p.adxl_az      = adxl_az;

  if (LOG_PORT == (Stream *)&BT_SERIAL) {
    muxSendFrame(MUX_TEL, (const uint8_t *)&p, (uint16_t)sizeof(p));
  }
}

// ============================================================
//  COMMAND HANDLER  (unchanged)
// ============================================================
void handleStream(Stream &port) {
  String line;
  if (!readLineFrom(port, line)) return;
  if (LOG_ON && !line.startsWith("log")) {
    LOG_ON = false; logHeaderPrinted = false; lastLogMs = 0;
    port.println("\r\n[log auto-stopped]");
  }
  if (line.startsWith("log")) {
    String rest = line.substring(3); rest.trim();
    if (rest.length() == 0) { port.println("\r\nUsage: log 1 label  |  log 0"); return; }
    int sp    = rest.indexOf(' ');
    String tokOn = (sp < 0) ? rest : rest.substring(0, sp); tokOn.trim();
    int on = tokOn.toInt();
    if (on == 0) { LOG_ON = false; logHeaderPrinted = false; port.println("\r\n[log off]"); return; }
    LOG_ON = true; LOG_PORT = &port; logHeaderPrinted = false;
    if (sp >= 0) {
      String lab = rest.substring(sp + 1); lab.trim();
      if (lab.length() > 0) lab.toCharArray(LOG_LABEL, sizeof(LOG_LABEL));
    }
    port.print("\r\n[log on] label="); port.println(LOG_LABEL);
    return;
  }
  if (line.charAt(0) == 'g') {
    String rest = line.substring(1); rest.trim();
    int s1 = rest.indexOf(' ');
    if (s1 < 0) { port.println("\r\nUsage: g x y theta"); return; }
    String tok1 = rest.substring(0, s1); tok1.trim();
    String rest2 = rest.substring(s1 + 1); rest2.trim();
    int s2 = rest2.indexOf(' ');
    if (s2 < 0) { port.println("\r\nUsage: g x y theta"); return; }
    String tok2 = rest2.substring(0, s2); tok2.trim();
    String tok3 = rest2.substring(s2 + 1); tok3.trim();
    planGoto(tok1.toFloat(), tok2.toFloat(), tok3.toFloat());
    LOG_ON = true; LOG_PORT = &port; logHeaderPrinted = false;
    strcpy(LOG_LABEL, "goto");
    autoGotoLogging = true; autoLogStartMs = millis();
    port.println("\r\n[goto accepted]");
    return;
  }
  if (line == "stop") {
    phase = IDLE; moving = false; rotating = false;
    stopAll(); imuVelocityProxyUpdate(true);
    port.println("\r\n[stopped]");
    return;
  }
  if (line == "z") {
    zeroYawToCurrent();
    curr_x_mm = curr_y_mm = curr_theta_deg = 0;
    phase = IDLE; moving = rotating = false;
    stopAll(); vx_imu = vy_imu = 0.0f;
    port.println("\r\n[reset]");
    return;
  }
  if (line == "s") {
    port.print("\r\n[status] phase="); port.print((int)phase);
    port.print(" ax=");    port.print(alpha_x_use, 3);
    port.print(" ay=");    port.print(alpha_y_use, 3);
    port.print(" I_sum="); port.print(i_sum_mA,    1);
    port.print(" I_dlt="); port.print(i_delta_mA,  1);
    port.print(" adxl=");  port.print(adxl_ax); port.print(",");
                           port.print(adxl_ay); port.print(",");
                           port.println(adxl_az);
    return;
  }
  port.println("\r\nCommands: g x y th | stop | z | s | log 1 label | log 0");
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  ESP_SERIAL.begin(ESP_BAUD);
  Wire.begin();
  Serial.println("\n=== Omni base Rev2 (BNO08x + ADXL345) ===");

  // --- BNO08x ---
  Serial.println("Starting BNO08x...");
  if (!bno08x.begin_I2C(BNO08X_ADDR)) {
    Serial.println("BNO08x not detected at 0x4B!");
    while (1);
  }
  // Enable game rotation vector (yaw, no magnetometer) and linear accel at 5 ms (200 Hz)
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 5000)) {
    Serial.println("Could not enable SH2_GAME_ROTATION_VECTOR");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 5000)) {
    Serial.println("Could not enable SH2_LINEAR_ACCELERATION");
  }
  delay(200);
  // Warm up: drain a few samples before zeroing
  for (int i = 0; i < 50; i++) { bno08x_poll(); delay(5); }
  zeroYawToCurrent();
  Serial.print("BNO08x OK. Yaw zeroed at "); Serial.print(bno_yaw_deg_raw, 2); Serial.println(" deg");

  // --- INA260 ---
  Serial.println("Starting INA260 sensors...");
  if (!ina_m3.begin(0x40)) { Serial.println("INA260 M3 (0x40) not detected!"); while (1); }
  if (!ina_m2.begin(0x41)) { Serial.println("INA260 M2 (0x41) not detected!"); while (1); }
  Serial.println("INA260 OK.");

  // --- ADXL345 ---
  Serial.println("Starting ADXL345...");
  adxl_ok = adxl345_begin();
  if (adxl_ok) {
    Serial.println("ADXL345 OK (0x53, ±2g, 100 Hz).");
  } else {
    Serial.println("ADXL345 not detected at 0x53 — telemetry fields will be zero.");
  }

  // --- Encoders ---
  pinMode(ENC1_A, INPUT_PULLUP); pinMode(ENC1_B, INPUT_PULLUP);
  pinMode(ENC2_A, INPUT_PULLUP); pinMode(ENC2_B, INPUT_PULLUP);
  pinMode(ENC3_A, INPUT_PULLUP); pinMode(ENC3_B, INPUT_PULLUP);

  counts_per_mm = COUNTS_PER_REV / (2.0f * PI * WHEEL_RADIUS_MM);
  stopAll();
  phase = IDLE; moving = false; rotating = false; last_us = 0;

  Serial.print("Ready. counts_per_mm="); Serial.println(counts_per_mm, 3);
  Serial.print("TelemetryPacket size="); Serial.print(sizeof(TelemetryPacket)); Serial.println(" bytes (expect 54)");
  Serial.println("Commands: g x y th | stop | z | s | log 1 label | log 0");
  Serial.println("XIAO link on Serial3 (pins 14/15), framed mux protocol.");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  static unsigned long lastPrint = 0;

  // 1. Poll sensors (non-blocking)
  bno08x_poll();                                       // update cached yaw + linear accel
  if (adxl_ok) adxl345_read(adxl_ax, adxl_ay, adxl_az); // update ADXL345 raw counts

  // 2. Communication
  muxPoll();
  handleStream(Serial);
  handleStream(BT_SERIAL);

  // 3. Slip estimator + currents
  slipEstimatorUpdate(true);
  readCurrents();

  // 4. Motion control
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
        BT_SERIAL.println("[done]");
      }
      break;
    case IDLE:
    default:
      imuVelocityProxyUpdate(true);
      break;
  }

  // 6. Telemetry emit
  logMaybePrint();

  // 7. Auto-stop logger after timeout
  if (autoGotoLogging && millis() - autoLogStartMs > AUTO_LOG_DURATION_MS) {
    LOG_ON = false; autoGotoLogging = false; logHeaderPrinted = false;
    Serial.println("[auto log stopped]");
    BT_SERIAL.println("[auto log stopped]");
  }
}