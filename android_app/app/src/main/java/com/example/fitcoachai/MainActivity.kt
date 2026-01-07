package com.example.fitcoachai

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.TextView
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var totalReps = 0
    private var correctReps = 0
    private var incorrectReps = 0

    private var lastRepTime: Long = 0

    private val minRepIntervalMs = 1200L


    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var txtStatus: TextView
    private lateinit var txtConfidence: TextView

    private lateinit var txtTotalReps: TextView
    private lateinit var txtCorrectValue: TextView
    private lateinit var txtIncorrectValue: TextView




    data class Sample(
        val t: Double,
        val ax: Double,
        val ay: Double,
        val az: Double,
        val gx: Double,
        val gy: Double,
        val gz: Double
    )

    private val sampleBuffer = ArrayList<Sample>()
    private var startTimeNs: Long = 0L
    private val windowSeconds = 3.0
    private val minSamples = 50
    private val handler = Handler(Looper.getMainLooper())
    private var isTracking = false

    // latest sensor values
    private var lastAx = 0.0
    private var lastAy = 0.0
    private var lastAz = 0.0
    private var lastGx = 0.0
    private var lastGy = 0.0
    private var lastGz = 0.0

    // HTTP
    private val client = OkHttpClient()
    // TODO: change to your real laptop IP (not 127.0.0.1)
    private val serverUrl = "http://0.0.0.00:8000"
    private val motionThreshold = 3.0f  //used for sending data only when having enough movements.
    private val sendWindowRunnable = object : Runnable {
        override fun run() {
            if (!isTracking) return
            Log.d("FitCoachAI", "Runnable tick, buffer size = ${sampleBuffer.size}")
            sendCurrentWindow()
            handler.postDelayed(this, 1500L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Views
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        txtStatus = findViewById(R.id.txtStatus)
        txtConfidence = findViewById(R.id.txtConfidence)

        txtTotalReps = findViewById(R.id.txtRepsValue)
        txtCorrectValue = findViewById(R.id.txtCorrectValue)
        txtIncorrectValue = findViewById(R.id.txtIncorrectValue)

        // Sensors
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // Clicks
        btnStart.setOnClickListener { startTracking() }
        btnStop.setOnClickListener { stopTracking() }

        // Init counters
        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        txtTotalReps.text = "0"
        txtCorrectValue.text = "0"
        txtIncorrectValue.text = "0"
    }



    private fun startTracking() {
        if (isTracking) return
        isTracking = true
        sampleBuffer.clear()
        startTimeNs = 0L

        accelSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        txtStatus.text = "Collecting..."
        txtConfidence.text = "Confidence —"
        Log.d("FitCoachAI", "Start tracking")

        handler.postDelayed(sendWindowRunnable, 1000L)
    }

    private fun stopTracking() {
        if (!isTracking) return
        isTracking = false
        sensorManager.unregisterListener(this)
        handler.removeCallbacks(sendWindowRunnable)

        txtStatus.text = "Stopped"
        txtConfidence.text = ""
        Log.d("FitCoachAI", "Stop tracking")
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isTracking) return

        if (startTimeNs == 0L) {
            startTimeNs = event.timestamp
        }
        val tSec = (event.timestamp - startTimeNs) / 1_000_000_000.0

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0].toDouble()
                lastAy = event.values[1].toDouble()
                lastAz = event.values[2].toDouble()
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0].toDouble()
                lastGy = event.values[1].toDouble()
                lastGz = event.values[2].toDouble()
            }
            else -> return
        }

        val sample = Sample(tSec, lastAx, lastAy, lastAz, lastGx, lastGy, lastGz)
        sampleBuffer.add(sample)

        val minTime = tSec - windowSeconds
        while (sampleBuffer.isNotEmpty() && sampleBuffer.first().t < minTime) {
            sampleBuffer.removeAt(0)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // not used
    }

    private fun sendCurrentWindow() {
        // 1) Check if we have enough samples for (about) one rep
        if (sampleBuffer.size < minSamples) {
            Log.d("FitCoachAI", "window too small (${sampleBuffer.size}), skipping")
            return
        }
      //   1) Check if there is enough motion in this window
        var minMag = Float.MAX_VALUE
        var maxMag = -Float.MAX_VALUE
        for (s in sampleBuffer) {
            val mag = kotlin.math.sqrt(
                (s.ax * s.ax + s.ay * s.ay + s.az * s.az).toDouble()
            ).toFloat()
            if (mag < minMag) minMag = mag
            if (mag > maxMag) maxMag = mag
        }
        val deltaMag = maxMag - minMag
        Log.d("FitCoachAI", "deltaMag for window = $deltaMag")
        val deltaThreshold = 1.0f   // start here, tune
        if (deltaMag < deltaThreshold) {
            Log.d("FitCoachAI", "idle/low‑motion window (deltaMag=$deltaMag), skipping")
            return
        }


        // 2) Simple time-based debounce so we don't double-count the same rep
        val now = System.currentTimeMillis()
        if (now - lastRepTime < minRepIntervalMs) {
            Log.d("FitCoachAI", "rep too soon (${now - lastRepTime} ms), skipping")
            return
        }

        // 3) Build JSON body from current window
        val jsonArray = JSONArray()
        for (sample in sampleBuffer) {
            val obj = JSONObject().apply {
                put("t", sample.t)
                put("ax", sample.ax)
                put("ay", sample.ay)
                put("az", sample.az)
                put("gx", sample.gx)
                put("gy", sample.gy)
                put("gz", sample.gz)
            }
            jsonArray.put(obj)
        }
        val jsonBody = JSONObject().apply {
            put("window", jsonArray)
        }

        val body = jsonBody.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$serverUrl/classify_window")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("FitCoachAI", "HTTP failure: ${e.message}", e)
                runOnUiThread {
                    txtStatus.text = "Server error"
                    txtConfidence.text = ""
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        Log.e("FitCoachAI", "Unexpected code $response")
                        runOnUiThread {
                            txtStatus.text = "Error ${response.code}"
                            txtConfidence.text = ""
                        }
                        return
                    }

                    val bodyString = response.body?.string() ?: ""
                    Log.d("FitCoachAI", "Response: $bodyString")

                    try {
                        val json = JSONObject(bodyString)
                        val label = json.getString("label")
                        val confidence = json.getDouble("confidence")

                        runOnUiThread {
                            if (confidence < 0.5) {   // tune threshold
                                Log.d("FitCoachAI", "low confidence ($confidence), not counting rep")
                                return@runOnUiThread
                            }
                            // 4) Update last rep time (we accept this rep)
                            lastRepTime = System.currentTimeMillis()

                            // Show label + confidence
                            txtStatus.text = label
                            txtConfidence.text = "Confidence: ${"%.2f".format(confidence)}"

                            // 5) Update counters
                            totalReps += 1
                            if (label.uppercase().startsWith("CORRECT") ||
                                (label.uppercase().startsWith("INCORRECT") && confidence < 0.6)
                            ) {
                                correctReps += 1
                            } else {
                                incorrectReps += 1
                            }


                            txtTotalReps.text = totalReps.toString()
                            txtCorrectValue.text = correctReps.toString()
                            txtIncorrectValue.text = incorrectReps.toString()
                        }

                        // 6) Clear buffer so next rep starts fresh
                        sampleBuffer.clear()
                    } catch (ex: Exception) {
                        Log.e("FitCoachAI", "Error parsing response", ex)
                        runOnUiThread {
                            txtStatus.text = "Parse error"
                            txtConfidence.text = ""
                        }
                    }
                }
            }
        })
    }
}

