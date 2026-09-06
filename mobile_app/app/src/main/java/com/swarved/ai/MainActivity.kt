package com.swarved.ai

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.swarved.ai.adapter.IncidentAdapter
import com.swarved.ai.model.ScamIncident

class MainActivity : AppCompatActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)


        val recyclerView =
            findViewById<RecyclerView>(
                R.id.recyclerIncidents
            )


        val incidents = listOf(

            ScamIncident(
                "Today • 4:32 PM",
                "Unknown Caller",
                96
            ),

            ScamIncident(
                "Today • 1:18 PM",
                "+91 98XXXXXX12",
                89
            ),

            ScamIncident(
                "Yesterday • 8:45 PM",
                "Unknown Caller",
                94
            )

        )


        recyclerView.layoutManager =
            LinearLayoutManager(this)

        recyclerView.adapter =
            IncidentAdapter(incidents)
    }
}
