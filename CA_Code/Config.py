class Config:
    #This is the name of the column where the full conversation or message text is stored.
    INTERACTION_CONTENT = "Interaction content"

    #This is the name of the column that contains a short summary or title of the ticket.
    TICKET_SUMMARY = "Ticket Summary"

    #This column is used to define what we want to predict.
    #We can change it to "Tone", "Resolution", or "Other" if needed.
    CLASS_COL = "Intent"

    #This column helps us divide the data into different groups, like by email address or user.
    GROUPED = "Mailbox"
