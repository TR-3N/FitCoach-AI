package com.example.fitcoachai

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
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

    private val sampleBuffer = mutableListOf<Sample>()
    private var isTracking = false
    private var startTimeNs: Long = 0L

    private val windowSeconds = 1.5f
    private val stepMillis = 1000L

    private val handler = Handler(Looper.getMainLooper())

    // CHANGE to your laptop IP
    private val serverUrl = "http://192.168.29.32:8000/classify_window"

    private val client: OkHttpClient by lazy { OkHttpClient.Builder().build() }

    data class Sample(
        val time: Float,
        val ax: Float,
        val ay: Float,
        val az: Float,
        val gx: Float,
        val gy: Float,
        val gz: Float
    )

    private val sendWindowRunnable = object : Runnable {
        override fun run() {
            if (!isTracking) return
            println("DEBUG: runnable tick")
            sendCurrentWindow()
            handler.postDelayed(this, stepMillis)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        txtStatus = findViewById(R.id.txtStatus)
        txtConfidence = findViewById(R.id.txtConfidence)

        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        btnStart.setOnClickListener {
            println("DEBUG: Start button clicked")
            startTracking()
        }
        btnStop.setOnClickListener {
            println("DEBUG: Stop button clicked")
            stopTracking()
        }
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

        txtStatus.text = "Preparing..."
        txtStatus.setTextColor(0xFFFFFFFF.toInt())
        txtConfidence.text = "Confidence —"

        handler.postDelayed(sendWindowRunnable, stepMillis)
    }

    private fun stopTracking() {
        if (!isTracking) return
        isTracking = false
        sensorManager.unregisterListener(this)
        handler.removeCallbacks(sendWindowRunnable)
        txtStatus.text = "Idle"
        txtStatus.setTextColor(0xFFFFFFFF.toInt())
        txtConfidence.text = "Confidence —"
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isTracking || event == null) return

        if (startTimeNs == 0L) startTimeNs = event.timestamp
        val tSec = (event.timestamp - startTimeNs) / 1_000_000_000.0f

        val ax: Float
        val ay: Float
        val az: Float
        val gx: Float
        val gy: Float
        val gz: Float

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                ax = event.values[0]
                ay = event.values[1]
                az = event.values[2]
                gx = 0f; gy = 0f; gz = 0f
            }
            Sensor.TYPE_GYROSCOPE -> {
                gx = event.values[0]
                gy = event.values[1]
                gz = event.values[2]
                ax = 0f; ay = 0f; az = 0f
            }
            else -> return
        }

        sampleBuffer.add(Sample(tSec, ax, ay, az, gx, gy, gz))

        val minTime = tSec - windowSeconds
        while (sampleBuffer.isNotEmpty() && sampleBuffer.first().time < minTime) {
            sampleBuffer.removeAt(0)
        }

        println("DEBUG: sensor sample added, buffer size = ${sampleBuffer.size}")
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun sendCurrentWindow() {
        println("DEBUG: sendCurrentWindow called, buffer size = ${sampleBuffer.size}")
        if (sampleBuffer.isEmpty()) return

        val samplesJson = JSONArray()
        for (s in sampleBuffer) {
            val obj = JSONObject()
            obj.put("time", s.time.toDouble())
            obj.put("ax", s.ax.toDouble())
            obj.put("ay", s.ay.toDouble())
            obj.put("az", s.az.toDouble())
            obj.put("gx", s.gx.toDouble())
            obj.put("gy", s.gy.toDouble())
            obj.put("gz", s.gz.toDouble())
            samplesJson.put(obj)
        }

        val root = JSONObject()
        root.put("samples", samplesJson)

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = root.toString().toRequestBody(mediaType)

        val request = Request.Builder()
            .url(serverUrl)
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    txtStatus.text = "Server error"
                    txtStatus.setTextColor(0xFFFFC107.toInt())
                    txtConfidence.text = ""
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    val respBody = it.body?.string() ?: return
                    try {
                        val obj = JSONObject(respBody)
                        val label = obj.optString("label", "UNKNOWN")
                        val conf = obj.optDouble("confidence", 0.0)
                        val confPct = (conf * 100).toInt()

                        runOnUiThread {
                            txtStatus.text = label
                            if (label == "CORRECT") {
                                txtStatus.setTextColor(0xFF4CAF50.toInt())
                            } else if (label == "INCORRECT") {
                                txtStatus.setTextColor(0xFFF44336.toInt())
                            } else {
                                txtStatus.setTextColor(0xFFFFFFFF.toInt())
                            }
                            txtConfidence.text = "Confidence $confPct%"
                        }
                    } catch (ex: Exception) {
                        runOnUiThread {
                            txtStatus.text = "Parse error"
                            txtStatus.setTextColor(0xFFFFC107.toInt())
                            txtConfidence.text = ""
                        }
                    }
                }
            }
        })
    }
}
