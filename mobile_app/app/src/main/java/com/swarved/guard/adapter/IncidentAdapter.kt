package com.swarved.guard.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.swarved.guard.R
import com.swarved.guard.model.ScamIncident

class IncidentAdapter(
    private val incidents: List<ScamIncident>
) : RecyclerView.Adapter<IncidentAdapter.IncidentViewHolder>() {

    class IncidentViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val callerId: TextView = view.findViewById(R.id.tvCallerId)
        val timestamp: TextView = view.findViewById(R.id.tvTimestamp)
        val probability: TextView = view.findViewById(R.id.tvProbability)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): IncidentViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_incident, parent, false)
        return IncidentViewHolder(view)
    }

    override fun onBindViewHolder(holder: IncidentViewHolder, position: Int) {
        val incident = incidents[position]
        holder.callerId.text = incident.callerId
        holder.timestamp.text = incident.timestamp
        holder.probability.text = "${incident.syntheticProbability}%"
    }

    override fun getItemCount(): Int = incidents.size
}
