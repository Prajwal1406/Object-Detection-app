package com.example.object_detect

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.FileUtils
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import com.example.object_detect.ml.AutoModel1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {
    private val paint = Paint()
    private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var bitmap: Bitmap
    private lateinit var model: AutoModel1
    private lateinit var labels:List<String>
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300,300,ResizeOp.ResizeMethod.BILINEAR)).build()

    private var colors = listOf<Int>(
        Color.BLUE,Color.GREEN,Color.RED,Color.CYAN,Color.GRAY,Color.BLACK,
        Color.DKGRAY,Color.MAGENTA,Color.YELLOW,Color.WHITE
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        labels = FileUtil.loadLabels(this,"ml.txt")
        model = AutoModel1.newInstance(this)
        imageView = findViewById(R.id.imageV)
        button = findViewById(R.id.btn)

        button.setOnClickListener(){
            startActivityForResult(intent,101)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == 101)
        {
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            get_prediction()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

            model.close()

    }

    fun get_prediction(){

        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray
        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888,true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width
        paint.textSize = h/35f
        paint.strokeWidth = h/85f
        var x = 0
        scores.forEachIndexed{ index,fl ->
            x = index
            x *=4
            if(fl >0.5){
                paint.color = colors[index]
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(
                    locations[x+1] *w,
                    locations[x] *h, locations[x+3] *w, locations[x+2] *h), paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels[classes[index].toInt()] +" "+fl.toString(),
                    locations[x+1] *w, locations[x] *h,paint)
            }
        }
        imageView.setImageBitmap(mutable)

    }
}