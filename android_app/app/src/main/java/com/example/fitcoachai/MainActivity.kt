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

    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var txtStatus: TextView
    private lateinit var txtConfidence: TextView

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
    private val serverUrl = "http://192.168.29.32:8000/classify_window"

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

        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        txtStatus = findViewById(R.id.txtStatus)
        txtConfidence = findViewById(R.id.txtConfidence)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        btnStart.setOnClickListener { startTracking() }
        btnStop.setOnClickListener { stopTracking() }
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
        if (sampleBuffer.size < minSamples) {
            Log.d("FitCoachAI", "window too small (${sampleBuffer.size}), skipping")
            return
        }

        Log.d("FitCoachAI", "sendCurrentWindow called, sending ${sampleBuffer.size} samples")

        val samplesArray = JSONArray()
        for (s in sampleBuffer) {
            val obj = JSONObject()
            obj.put("t", s.t)
            obj.put("ax", s.ax)
            obj.put("ay", s.ay)
            obj.put("az", s.az)
            obj.put("gx", s.gx)
            obj.put("gy", s.gy)
            obj.put("gz", s.gz)
            samplesArray.put(obj)
        }

        val root = JSONObject()
        root.put("window", samplesArray)   // must match WindowIn.window

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = root.toString().toRequestBody(mediaType)

        val request = Request.Builder()
            .url(serverUrl)
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("FitCoachAI", "HTTP failure: ${e.message}")
                runOnUiThread {
                    txtStatus.text = "Server error"
                    txtConfidence.text = ""
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        Log.e("FitCoachAI", "HTTP error: ${response.code}")
                        runOnUiThread {
                            txtStatus.text = "Server error"
                            txtConfidence.text = ""
                        }
                        return
                    }

                    val bodyStr = response.body?.string() ?: ""
                    Log.d("FitCoachAI", "HTTP response: $bodyStr")

                    try {
                        val json = JSONObject(bodyStr)
                        val label = json.optString("label", "UNKNOWN")
                        val confidence = json.optDouble("confidence", 0.0)

                        // SUCCESS UI UPDATE (modify this one)
                        runOnUiThread {
                            txtStatus.text = label
                            val color = if (label == "CORRECT") 0xFF4CAF50.toInt() else 0xFFF44336.toInt()
                            txtStatus.setTextColor(color)
                            txtConfidence.text = "Confidence: ${"%.2f".format(confidence)}"
                        }
                    } catch (e: Exception) {
                        Log.e("FitCoachAI", "Parse error: ${e.message}")
                        // ERROR UI UPDATE (leave this as is)
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

